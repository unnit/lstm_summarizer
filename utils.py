import nltk
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('punkt')

SPECIAL_TOKENS = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}


def tokenize(text):
    return word_tokenize(text.lower())


def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))
    vocab = {word: i+len(SPECIAL_TOKENS) for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab.update(SPECIAL_TOKENS)
    return vocab


def vectorize(text, vocab, add_tokens=True):
    tokens = tokenize(text)
    if add_tokens:
        tokens = ['<sos>'] + tokens + ['<eos>']
    return torch.tensor([vocab.get(t, vocab['<unk>']) for t in tokens], dtype=torch.long)


def pad_sequences(sequences, pad_value=0):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_value)

def detokenize(indices, inv_vocab):
    words = []
    for idx in indices:
        word = inv_vocab.get(idx, '<unk>')
        if word in ['<sos>', '<eos>', '<pad>']:
            continue
        words.append(word)
    return ' '.join(words)
