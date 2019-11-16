import io
import progressbar
import argparse
import datetime
import random
import nltk.translate.bleu_score as bleu
import torch
from torch import nn as N
from torch import optim as O
from torch.nn import functional as F


BOS = 0
EOS = 1
UNK = 2
PAD = 3


class Encoder(N.Module):
    def __init__(self, vocab_size, embed_size, output_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.LSTM_layer = N.LSTM(embed_size, output_size, num_layers, batch_first=True)

    def forward(self, inputs):
        inputs = self.embedding_layer(inputs)
        _, (hn, _) = self.LSTM_layer(inputs)
        return hn


class Decoder(N.Module):
    def __init__(self, vocab_size, batch_size, embed_size, output_size, num_layers, device):
        super(Decoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.LSTM_layer = N.LSTM(embed_size, output_size, num_layers, batch_first=True)
        self.output_layer = N.Linear(output_size, vocab_size)
        self.c_train = torch.zeros(num_layers, batch_size, output_size).to(device)
        self.c_val = torch.zeros(num_layers, 1, output_size).to(device)

    def forward(self, xs, h, val=False):
        if not val:
            c = self.c_train
        else:
            c = self.c_val
        xs = self.embedding_layer(xs)
        output, (h, c) = self.LSTM_layer(xs, (h, c))
        output = self.output_layer(output)
        return output, h, c


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


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar(maxval=n_lines)
    data = []
    print('loading...: %s' % path)
    with io.open(path, encoding='utf-8') as f:
        for line in bar(f):
            words = line.strip().split()
            array = torch.LongTensor([vocabulary.get(w, UNK) for w in words])
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = 0
    for s in data:
        for w in s:
            if w == UNK:
                unknown += 1
    total = sum(s.size(0) for s in data)
    return unknown / total


def train_iterator(data, batch_size):
    for i in range(len(data)//batch_size):
        yield [data_batch[0] for data_batch in data[batch_size*i:batch_size*(i+1)]], \
              [data_batch[1] for data_batch in data[batch_size*i:batch_size*(i+1)]]


def main():
    parser = argparse.ArgumentParser(description='seq2seq by Pytorch')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source', help='source sentence list for validation')
    parser.add_argument('--validation-target', help='target sentence list for validation')
    parser.add_argument('--min-source-sentence', type=int, default=1, help='minimum length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=20, help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1, help='minimum length of target sentence')
    parser.add_argument('--max_target_sentence', type=int, default=20, help='maximum length of target sentence')
    parser.add_argument('--batch_size', type=int, default=6, help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', type=int, default=3, help='number of sweeps over the dataset to train')
    parser.add_argument('--interval', type=int, default=2, help='number of layers')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--embed_size', type=int, default=32, help='embed size')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--device', default="cuda", help='number of using device')
    args = parser.parse_args()
    # load train data set
    print('[{}] Loading data set... (this may take several minutes)'.format(datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    source_vocab_size = len(source_ids)
    target_vocab_size = len(target_ids)
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)
    train_source_unknown = calculate_unknown_ratio(train_source)
    train_target_unknown = calculate_unknown_ratio(train_target)
    assert len(train_source) == len(train_target), "source data size and target data size don't accord."
    print('[{}] Data set loaded.'.format(datetime.datetime.now()))
    # print information of train data
    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_source))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))
    train_data = [(s, t) for s, t in zip(train_source, train_target)]

    # make id to source dictionary
    source_words = {i: w for w, i in source_ids.items()}
    target_words = {i: w for w, i in target_ids.items()}

    # load validation data set if it exist
    if args.validation_source and args.validation_target:
        test_source = load_data(source_ids, args.validation_source)
        test_target = load_data(target_ids, args.validation_target)
        assert len(test_source) == len(test_target), 'validation data size is invalid'
        test_data = [(s, t) for s, t in zip(test_source, test_target)]
        print('Validation data: %d' % len(test_data))

    # assign vectors of bos and eos
    bos = torch.zeros(1, dtype=torch.int64)
    eos = torch.ones(1, dtype=torch.int64)
    # make instances of encoder and decoder
    encoder = Encoder(source_vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(args.device)
    decoder = Decoder(target_vocab_size, args.batch_size, args.embed_size, args.hidden_size, args.num_layers,
                      args.device).to(args.device)
    # define optimizer
    encoder_optimizer = O.Adam(encoder.parameters())
    decoder_optimizer = O.Adam(decoder.parameters())
    # define loss function
    criterion = N.CrossEntropyLoss(ignore_index=PAD)
    # start training
    for i in range(args.epoch):
        # shuffle train data set
        random.shuffle(train_data)
        step = 0
        iterator = train_iterator(train_data, args.batch_size)
        for sources_batch, targets_batch in iterator:
            assert len(sources_batch) == args.batch_size, "source train data's batch size is invalid"
            assert len(targets_batch) == args.batch_size, "target train data's batch size is invalid"
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            decoder_source = [torch.cat((bos, sentence), dim=0) for sentence in targets_batch]
            decoder_target = [torch.cat((sentence, eos), dim=0) for sentence in targets_batch]
            sources_batch = N.utils.rnn.pad_sequence(sources_batch, batch_first=True, padding_value=PAD)
            decoder_source = N.utils.rnn.pad_sequence(decoder_source, batch_first=True, padding_value=PAD)
            decoder_target = N.utils.rnn.pad_sequence(decoder_target, batch_first=True, padding_value=PAD)
            hn = encoder(sources_batch.to(args.device))
            output, _, _ = decoder(decoder_source.to(args.device), hn)
            loss = 0
            for o, t in zip(output, decoder_target.to(args.device)):
                loss += criterion(o, t)
            print("epoch: {},step: {}, loss = {}".format(i, step, float(loss)))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            step += 1
            if step % args.interval == 0 and args.validation_source and args.validation_target:
                with torch.no_grad():
                    encoder.eval()
                    decoder.eval()
                    random.shuffle(test_data)
                    source, target = test_data[0][0].reshape(1, test_data[0][0].size(0)), test_data[0][1]
                    hn = encoder(source.to(args.device))
                    token = '<BOS>'
                    result = []
                    while True:
                        word = target_ids[token]
                        output, hn, c = decoder(torch.LongTensor([[word]]).to(args.device), hn, val=True)
                        prob = F.softmax(torch.squeeze(output))
                        index = torch.argmax(prob).item()
                        token = target_words[index]
                        if token == "<EOS>":
                            break
                        result.append(token)
                        if len(result) > args.max_target_sentence:
                            break
                    source = torch.squeeze(source)
                    print(source)
                    source_sentence = ' '.join([source_words[int(x)] for x in source])
                    target_sentence = ' '.join([target_words[int(y)] for y in target])
                    result_sentence = ' '.join([y for y in result])
                    print('# source : ' + source_sentence)
                    print('# result : ' + result_sentence)
                    print('# expect : ' + target_sentence)
                    score = bleu.sentence_bleu([target_sentence], result_sentence)
                    print('BLEUScore : {:.1f}'.format(score*100))


if __name__ == '__main__':
    main()
