#!/usr/bin/env python
"""
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
import pickle
import re


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word: str) -> int:
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word: str) -> bool:
        return word in self.word2id

    def __setitem__(self, key: str, value: int):
        raise ValueError('vocabulary is readonly')

    def __len__(self) -> int:
        return len(self.word2id)

    def __repr__(self) -> str:
        return 'Vocabulary[size=%d]' % len(self)

    def get_word(self, wid: int) -> str:
        return self.id2word[wid]

    def add(self, word: str) -> int:
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents: List[str]) -> List[int]:
        return [self[w] for w in sents]
      
    @staticmethod
    def from_corpus(corpus: List[List[str]], size: int, freq_cutoff: int = 2) -> 'VocabEntry':
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry
