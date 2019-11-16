import io
import progressbar
import argparse
import datetime
import torch

BOS = 0
EOS = 1
UNK = 2
PAD = 3


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
            array = [vocabulary.get(w, UNK) for w in words]
            data.append(array)
    return data


def main():
    parser = argparse.ArgumentParser(description='seq2seq by Pytorch')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimum length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimum length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    args = parser.parse_args()

    print('[{}] Loading dataset... (this may take several minutes)'.format(datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)
    assert len(train_source) == len(train_target), "source data size and target data size don't accord."
    source_max_len = 0
    target_max_len = 0
    for i in range(len(train_source)):
        if source_max_len < len(train_source[i]):
            source_max_len = len(train_source[i])
        if target_max_len < len(train_target[i]):
            target_max_len = len(train_target[i])
    for i in range(len(train_source)):
        if len(train_source[i]) < source_max_len:
            train_source[i].extend([source_ids['<PAD>']]*(source_max_len - len(train_source[i])))
        if len(train_target[i]) < target_max_len:
            train_target[i].extend([target_ids['<PAD>']]*(target_max_len - len(train_target[i])))
    source_words = {i: w for w, i in source_ids.items()}
    target_words = {i: w for w, i in target_ids.items()}
    source_train_data = torch.LongTensor(train_source)
    target_train_data = torch.LongTensor(train_target)
    print(source_train_data)
    print(target_train_data)




if __name__ == '__main__':
    main()
