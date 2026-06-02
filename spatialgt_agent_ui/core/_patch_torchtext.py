"""
Provide a minimal shim for torchtext.vocab so that gene_tokenizer.py
works without installing the real torchtext package (which has
persistent ABI / C++ extension issues with different PyTorch builds).

The shim re-implements only the two symbols gene_tokenizer.py needs:
  - torchtext.vocab.Vocab
  - torchtext.vocab.vocab(ordered_dict, min_freq)

Import this module (and call patch()) before any code that touches torchtext.
"""

import sys
import types
from collections import OrderedDict
from typing import Dict, List, Optional


# ── Minimal Vocab reimplementation ───────────────────────────────────────────

class Vocab:
    """Minimal reimplementation of torchtext.vocab.Vocab."""

    def __init__(self, vocab_obj=None):
        """
        Args:
            vocab_obj: OrderedDict mapping token->freq, or dict mapping token->index.
                       The real torchtext Vocab.__init__ takes an internal torch Vocab
                       object, but GeneVocab calls super().__init__(ordered_dict)
                       where ordered_dict is what .vocab returns.
        """
        self._stoi: Dict[str, int] = {}
        self._itos: List[str] = []
        self._default_index: int = -1
        if vocab_obj is not None and isinstance(vocab_obj, dict):
            for token in vocab_obj:
                if token not in self._stoi:
                    self._stoi[token] = len(self._itos)
                    self._itos.append(token)

    @property
    def vocab(self) -> OrderedDict:
        """Return OrderedDict[token, index] — mirrors real torchtext Vocab.vocab"""
        return OrderedDict((t, i) for t, i in self._stoi.items())

    def __getitem__(self, token: str) -> int:
        return self._stoi.get(token, self._default_index)

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        return list(self._itos)

    def set_default_index(self, index: int) -> None:
        self._default_index = index

    def insert_token(self, token: str, index: int) -> None:
        if token in self._stoi:
            return
        # Fast path: appending at the end (most common in from_dict sorted loading)
        if index >= len(self._itos):
            self._stoi[token] = len(self._itos)
            self._itos.append(token)
        else:
            self._itos.insert(index, token)
            # Only rebuild stoi for tokens shifted by the insert
            for i in range(index, len(self._itos)):
                self._stoi[self._itos[i]] = i

    def append_token(self, token: str) -> None:
        if token in self._stoi:
            return
        self._stoi[token] = len(self._itos)
        self._itos.append(token)

    def lookup_token(self, index: int) -> str:
        return self._itos[index]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[t] for t in tokens]


def vocab(ordered_dict: OrderedDict, min_freq: int = 1) -> Vocab:
    """Factory matching torchtext.vocab.vocab(ordered_dict, min_freq)."""
    filtered = OrderedDict()
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            filtered[token] = freq
    return Vocab(filtered)


# ── Patch function ───────────────────────────────────────────────────────────

def patch():
    """
    Register shim modules so that `import torchtext.vocab` and
    `from torchtext.vocab import Vocab` resolve to our shim
    without the real torchtext package being installed.
    """
    if "torchtext" in sys.modules:
        return

    # torchtext top-level
    tt = types.ModuleType("torchtext")
    tt.__path__ = []
    sys.modules["torchtext"] = tt

    # torchtext._extension (in case something tries to import it)
    ext = types.ModuleType("torchtext._extension")
    ext._init_extension = lambda: None
    sys.modules["torchtext._extension"] = ext
    tt._extension = ext

    # torchtext.vocab
    vmod = types.ModuleType("torchtext.vocab")
    vmod.Vocab = Vocab
    vmod.vocab = vocab
    sys.modules["torchtext.vocab"] = vmod
    tt.vocab = vmod


patch()


patch()
