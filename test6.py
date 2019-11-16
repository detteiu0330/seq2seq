import torch
from torch import nn as N
from torch import optim as O
from torch.utils import data as D


class Encoder(N.Module):
    def __init__(self, vocab_size, embed_size, output_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size + 2, embed_size)
        self.LSTM_layer = N.LSTM(embed_size, output_size, num_layers, batch_first=True)

    def forward(self, inputs):
        inputs = self.embedding_layer(inputs)
        _, (hn, _) = self.LSTM_layer(inputs)
        return hn


class Decoder(N.Module):
    def __init__(self, vocab_size, batch_size, embed_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size + 2, embed_size)
        self.LSTM_layer = N.LSTM(embed_size, output_size, num_layers, batch_first=True)
        self.output_layer = N.Linear(output_size, vocab_size + 2)
        self.c = torch.zeros(num_layers, batch_size, output_size)

    def forward(self, xs, h):
        xs = self.embedding_layer(xs)
        output, (_, _) = self.LSTM_layer(xs, (h, self.c))
        output = self.output_layer(output)

        return output


def main():
    num_layers = 2
    len_seq = 10
    embed_size = 100
    hidden_size = 200
    num_sentences = 1024
    batch_size = 16
    vocab_size = 100
    bos = torch.zeros((batch_size, 1), dtype=torch.int64)
    eos = torch.ones((batch_size, 1), dtype=torch.int64)
    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers)
    decoder = Decoder(vocab_size, batch_size, embed_size, hidden_size, num_layers)
    encoder_optimizer = O.Adam(encoder.parameters())
    decoder_optimizer = O.Adam(decoder.parameters())
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    criterion = N.CrossEntropyLoss()
    sources = torch.randint(2, vocab_size + 2, (num_sentences, len_seq))
    targets = torch.randint(2, vocab_size + 2, (num_sentences, len_seq))
    train_data = D.TensorDataset(sources, targets)
    train_iterator = D.DataLoader(train_data, batch_size, shuffle=True)
    for i, train_data in enumerate(train_iterator):
        sources, targets = train_data
        xs = torch.cat((bos, targets), dim=1)
        ts = torch.cat((targets, eos), dim=1)
        hn = encoder(sources)
        output = decoder(xs, hn)
        loss = 0
        for o, t in zip(output, ts):
            loss += criterion(o, t)
        loss = loss/batch_size
        print("step {}: loss = {}".format(i, loss))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


if __name__ == '__main__':
    main()
