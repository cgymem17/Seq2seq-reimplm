from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
batch_size = 128
MAX_LENGTH = 10
hidden_size = 1000

# Some helper class and functions


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.numofwords = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.numofwords
            self.word2count[word] = 1
            self.index2word[self.numofwords] = word
            self.numofwords += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    """Read data. The format should be: lang1 senctences.    lang2 senctences."""
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse source and target langs if the reverse flag is True
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        inputLang = Lang(lang2)
        outputLang = Lang(lang1)
    else:
        inputLang = Lang(lang1)
        outputLang = Lang(lang2)

    return inputLang, outputLang, pairs


def prepareData(lang1, lang2, reverse=False):
    inputLang, outputLang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        inputLang.addSentence(pair[0])
        outputLang.addSentence(pair[1])
    print("Counted words:")
    print(inputLang.name, inputLang.numofwords)
    print(outputLang.name, outputLang.numofwords)
    return inputLang, outputLang, pairs


def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(inputLang, pair[0])
    target_tensor = tensorFromSentence(outputLang, pair[1])
    return (input_tensor, target_tensor)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# The Model class and training & evaluating functions
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)

        # Uniform Initialization
        self.embedding.weight.data.uniform_(-0.08, 0.08)
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # Uniform Initialization
        self.embedding.weight.data.uniform_(-0.08, 0.08)
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        self.out.weight.data.uniform_(-0.08, 0.08)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, n_half_epoch, print_every, learning_rate=0.7):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    criterion = nn.NLLLoss()
    n_iters_per_half_epoch = n_iters // n_half_epoch

    for half_epoch in range(n_half_epoch):
        # For every half epoch, half the lr after 5 epochs.
        if half_epoch > 10:
            learning_rate /= 2
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        for iter in range(1, n_iters_per_half_epoch + 1):
            training_pair = tensorsFromPair(random.choice(pairs))
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, 100 * (half_epoch + 1) * iter / n_iters, print_loss_avg))


def beam_search_decode(decoder, decoder_hidden, encoder_outputs, max_length=MAX_LENGTH, beam_size=2):
    candidates = [(decoder_hidden, [], 0.0)]
    for _ in range(max_length):
        temp_candidates = []
        for hidden, sequence, score in candidates:
            if sequence and sequence[-1] == EOS_token:
                temp_candidates.append((hidden, sequence, score))
            else:
                decoder_input = torch.tensor(
                    [sequence[-1]], device=device) if sequence else torch.tensor([SOS_token], device=device)
                decoder_output, hidden = decoder(decoder_input, hidden)
                top_scores, top_indices = torch.topk(
                    F.log_softmax(decoder_output, dim=1), beam_size)
                for i in range(beam_size):
                    next_score = top_scores[0][i].item()
                    next_index = top_indices[0][i].item()
                    temp_candidates.append(
                        (hidden, sequence + [next_index], score + next_score))
        temp_candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = temp_candidates[:beam_size]
    return candidates[0][1]


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(inputLang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_hidden = encoder_hidden
        decoded_words = beam_search_decode(
            decoder, decoder_hidden, encoder_outputs, max_length, beam_size=2)
        return [outputLang.index2word[index] for index in decoded_words if index != EOS_token]


def evaluateRandomly(encoder, decoder, n=10):
    """Randomly choose n pairs to evaluate."""
    bleu_scores = []
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

        # Calculate BLEU score
        # the reference sentence should be a list of words
        reference = [pair[1].split()]
        candidate = output_words  # the candidate sentence should be a list of words
        score = sentence_bleu(reference, candidate)
        print('BLEU Score:', score)
        print('')
        bleu_scores.append(score)

    bleu_score_avg = sum(bleu_scores) / len(bleu_scores)
    print('Average BLEU Score:', bleu_score_avg)


inputLang, outputLang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

encoder = EncoderRNN(inputLang.numofwords, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, outputLang.numofwords).to(device)

trainIters(encoder, decoder, n_iters=75000, n_half_epoch=15,
           print_every=5000, learning_rate=0.7)
evaluateRandomly(encoder, decoder)
