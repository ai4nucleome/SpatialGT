"""Small pure-Python vocabulary with stable gene-token indices.

The model uses JSON gene vocabularies with consecutive integer indices.
"""
from __future__ import annotations

from collections import OrderedDict


class Vocab:
    def __init__(self, vocab_obj=None):
        self._itos = []
        self._stoi = {}
        self._default_index = None
        if vocab_obj is not None:
            tokens = vocab_obj.get_itos() if hasattr(vocab_obj, "get_itos") else vocab_obj
            for token in tokens:
                self.append_token(token)

    @property
    def vocab(self):
        return self

    def __len__(self):
        return len(self._itos)

    def __contains__(self, token):
        return token in self._stoi

    def __getitem__(self, token):
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __call__(self, tokens):
        return self.lookup_indices(tokens)

    def get_stoi(self):
        return dict(self._stoi)

    def get_itos(self):
        return list(self._itos)

    def get_default_index(self):
        return self._default_index

    def set_default_index(self, index):
        self._default_index = index

    def insert_token(self, token, index):
        if token in self._stoi:
            raise ValueError("Token already exists: %s" % token)
        if not 0 <= index <= len(self._itos):
            raise ValueError("Vocabulary indices must be consecutive; got %s" % index)
        self._itos.insert(index, token)
        self._stoi = {token: i for i, token in enumerate(self._itos)}

    def append_token(self, token):
        if token in self._stoi:
            raise ValueError("Token already exists: %s" % token)
        self._stoi[token] = len(self._itos)
        self._itos.append(token)

    def lookup_token(self, index):
        return self._itos[index]

    def lookup_tokens(self, indices):
        return [self.lookup_token(i) for i in indices]

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]


def vocab(ordered_dict: OrderedDict, min_freq=1):
    return Vocab([token for token, count in ordered_dict.items() if count >= min_freq])
