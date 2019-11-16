import io
import progressbar
import argparse
import datetime

import torch
from torch import nn as N
from torch import optim as O
from torch.utils import data as D

BOS = 0
EOS = 1
UNK = 2
PAD = 3


class Encoder(N.Module):
    def __init__(self, vocab_size, embed_size, output_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size + 4, embed_size, padding_idx=PAD)
        self.LSTM_layer = N.LSTM(embed_size, output_size, num_layers, batch_first=True)

    def forward(self, inputs):
        inputs = self.embedding_layer(inputs)
        _, (hn, _) = self.LSTM_layer(inputs)
        return hn


class Decoder(N.Module):
    def __init__(self, vocab_size, batch_size, embed_size, output_size, num_layers, device):
        super(Decoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size + 4, embed_size, padding_idx=PAD)
        self.LSTM_layer = N.LSTM(embed_size, output_size, num_layers, batch_first=True)
        self.output_layer = N.Linear(output_size, vocab_size + 2)
        self.c = torch.zeros(num_layers, batch_size, output_size).to(device)

    def forward(self, xs, h):
        xs = self.embedding_layer(xs)
        output, (_, _) = self.LSTM_layer(xs, (h, self.c))
        output = self.output_layer(output)
        return output


def count_lines(path):
    with io.open(path, encoding='utf-8') as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with io.open(path, encoding='utf-8') as f:
        # +4 for BOS, EOS, UNK and PAD
        word_ids = {line.strip(): i + 4 for i, line in enumerate(f)}
    word_ids['<BOS>'] = 0
    word_ids['<EOS>'] = 1
    word_ids['<UNK>'] = 2
    word_ids['<PAD>'] = 3
    return word_ids


def load_data(vocabulary, path, device):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar(maxval=n_lines)
    data = []
    max_len = 0
    print('loading...: %s' % path)
    with io.open(path, encoding='utf-8') as f:
        for line in bar(f):
            words = line.strip().split()
            array = torch.LongTensor([vocabulary.get(w, UNK) for w in words]).to(device)
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    print(data)
    unknown = 0
    for s in data:
        for w in s:
            if w == UNK:
                unknown += 1
    total = sum(len(s) for s in data)
    print(unknown)
    print(total)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='seq2seq by Pytorch')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--min-source-sentence', type=int, default=1, help='minimum length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50, help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1, help='minimum length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50, help='maximum length of target sentence')
    parser.add_argument('--batch_size', type=int, default=16, help='number of sentence pairs in each mini-batch')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--embed_size', type=int, default=128, help='embed size')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--device', default="cuda", help='number of using device')
    args = parser.parse_args()

    print('[{}] Loading data set... (this may take several minutes)'.format(datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    source_vocab_size = len(source_ids)
    target_vocab_size = len(target_ids)
    train_source = load_data(source_ids, args.SOURCE, args.device)
    train_target = load_data(target_ids, args.TARGET, args.device)
    assert len(train_source) == len(train_target), "source data size and target data size don't accord."
    print('[{}] Data set loaded.'.format(datetime.datetime.now()))

    #train_source_unknown = calculate_unknown_ratio([s for s in train_source])
    #train_target_unknown = calculate_unknown_ratio([t for t in train_target])

    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_source))
    #print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    #print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    train_source = N.utils.rnn.pad_sequence(train_source, batch_first=True, padding_value=PAD)
    train_target = N.utils.rnn.pad_sequence(train_target, batch_first=True, padding_value=PAD)
    print(train_source)
    print(train_target)

    source_words = {i: w for w, i in source_ids.items()}
    target_words = {i: w for w, i in target_ids.items()}

    bos = torch.zeros((args.batch_size, 1), dtype=torch.int64).to(args.device)
    eos = torch.ones((args.batch_size, 1), dtype=torch.int64).to(args.device)

    encoder = Encoder(source_vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(args.device)
    decoder = Decoder(target_vocab_size, args.batch_size, args.embed_size, args.hidden_size, args.num_layers, args.device).to(args.device)

    encoder_optimizer = O.Adam(encoder.parameters())
    decoder_optimizer = O.Adam(decoder.parameters())

    #train_source = torch.LongTensor(train_source).to(args.device)
    #train_target = torch.LongTensor(train_target).to(args.device)

    criterion = N.CrossEntropyLoss(ignore_index=PAD)
    train_data = D.TensorDataset(train_source, train_target)
    train_iterator = D.DataLoader(train_data, args.batch_size, shuffle=True)

    for step, train_data in enumerate(train_iterator):
        if len(train_data) == args.batch_size:
            sources, targets = train_data
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            xs = torch.cat((bos, targets), dim=1)
            ts = torch.cat((targets, eos), dim=1)
            hn = encoder(sources)
            output = decoder(xs, hn)
            loss = 0
            for o, t in zip(output, ts):
                loss += criterion(o, t)
            print("step {}: loss = {}".format(step, loss))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()


if __name__ == '__main__':
    main()
