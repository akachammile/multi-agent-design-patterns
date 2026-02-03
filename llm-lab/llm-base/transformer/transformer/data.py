"""
Data Processing Module for Transformer Training

This module handles:
- Tokenization using spaCy
- Vocabulary building
- Dataset and DataLoader creation
- Batch processing with masking

Source Reference:
- Harvard NLP "The Annotated Transformer": https://github.com/harvardnlp/annotated-transformer
"""

import os
import pickle
from typing import List, Tuple, Optional, Iterator
from collections import Counter

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Using simple whitespace tokenization.")


# =============================================================================
# Tokenizers
# =============================================================================


class SimpleTokenizer:
    """Simple whitespace-based tokenizer (fallback when spaCy is not available)."""

    def __init__(self, language: str = "en"):
        self.language = language

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.strip().lower().split()

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class SpacyTokenizer:
    """Tokenizer using spaCy."""

    # Language model mapping
    LANG_MODELS = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "zh": "zh_core_web_sm",
    }

    def __init__(self, language: str = "en"):
        self.language = language
        model_name = self.LANG_MODELS.get(language, language)

        try:
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])
        except OSError:
            print(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
            raise

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return [tok.text.lower() for tok in self.nlp(text)]

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


def get_tokenizer(language: str = "en") -> SimpleTokenizer:
    """Get appropriate tokenizer for language."""
    if SPACY_AVAILABLE:
        try:
            return SpacyTokenizer(language)
        except OSError:
            pass
    return SimpleTokenizer(language)


# =============================================================================
# Vocabulary
# =============================================================================


class Vocabulary:
    """Vocabulary class for mapping tokens to indices."""

    def __init__(
        self,
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Initialize with special tokens
        self.token2idx = {
            pad_token: 0,
            sos_token: 1,
            eos_token: 2,
            unk_token: 3,
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.sos_token]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.eos_token]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]

    def __len__(self) -> int:
        return len(self.token2idx)

    def add_token(self, token: str) -> int:
        """Add a token to vocabulary. Returns the token index."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]

    def encode(self, tokens: List[str], add_special: bool = True) -> List[int]:
        """Convert tokens to indices, optionally adding SOS/EOS."""
        indices = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        if add_special:
            indices = [self.sos_idx] + indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """Convert indices to tokens."""
        tokens = [self.idx2token.get(i, self.unk_token) for i in indices]
        if remove_special:
            tokens = [
                t
                for t in tokens
                if t not in [self.pad_token, self.sos_token, self.eos_token]
            ]
        return tokens

    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "token2idx": self.token2idx,
                    "idx2token": self.idx2token,
                    "special_tokens": {
                        "pad": self.pad_token,
                        "sos": self.sos_token,
                        "eos": self.eos_token,
                        "unk": self.unk_token,
                    },
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        vocab = cls(
            pad_token=data["special_tokens"]["pad"],
            sos_token=data["special_tokens"]["sos"],
            eos_token=data["special_tokens"]["eos"],
            unk_token=data["special_tokens"]["unk"],
        )
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = data["idx2token"]
        return vocab


def build_vocabulary(data_path: str, tokenizer, min_freq: int = 2) -> Vocabulary:
    """Build vocabulary from a text file."""
    vocab = Vocabulary()
    counter = Counter()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenizer(line.strip())
            counter.update(tokens)

    # Add tokens that appear at least min_freq times
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab.add_token(token)

    return vocab


# =============================================================================
# Dataset
# =============================================================================


class TranslationDataset(Dataset):
    """Dataset for sequence-to-sequence translation."""

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        src_tokenizer,
        tgt_tokenizer,
        max_len: int = 256,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        # Load data
        self.src_data = []
        self.tgt_data = []

        with (
            open(src_path, "r", encoding="utf-8") as f_src,
            open(tgt_path, "r", encoding="utf-8") as f_tgt,
        ):
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_tokens = src_tokenizer(src_line.strip())
                tgt_tokens = tgt_tokenizer(tgt_line.strip())

                # Filter by length (accounting for SOS/EOS tokens)
                if len(src_tokens) <= max_len - 2 and len(tgt_tokens) <= max_len - 2:
                    self.src_data.append(src_vocab.encode(src_tokens))
                    self.tgt_data.append(tgt_vocab.encode(tgt_tokens))

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )


# =============================================================================
# Batch Processing
# =============================================================================


class Batch:
    """
    Object for holding a batch of data with masks.

    Attributes:
        src: Source sequence tensor (batch, src_len)
        src_mask: Source mask tensor (batch, 1, src_len)
        tgt: Target sequence tensor (batch, tgt_len)
        tgt_y: Target labels (shifted right) (batch, tgt_len-1)
        tgt_mask: Target mask tensor (batch, tgt_len, tgt_len)
        ntokens: Number of non-padding tokens in target
    """

    def __init__(self, src: Tensor, tgt: Optional[Tensor] = None, pad_idx: int = 0):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]  # Input to decoder (remove last token)
            self.tgt_y = tgt[:, 1:]  # Labels (remove first token - SOS)
            self.tgt_mask = self._make_std_mask(self.tgt, pad_idx)
            self.ntokens = (self.tgt_y != pad_idx).sum().item()
        else:
            self.tgt = None
            self.tgt_y = None
            self.tgt_mask = None
            self.ntokens = 0

    @staticmethod
    def _make_std_mask(tgt: Tensor, pad_idx: int) -> Tensor:
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def subsequent_mask(size: int) -> Tensor:
    """Create subsequent mask for decoder self-attention."""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.uint8)
    return mask == 0


def collate_fn(batch: List[Tuple[Tensor, Tensor]], pad_idx: int = 0) -> Batch:
    """Collate function for DataLoader."""
    src_batch, tgt_batch = zip(*batch)

    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return Batch(src_padded, tgt_padded, pad_idx)


# =============================================================================
# DataLoader Creation
# =============================================================================


def create_dataloaders(
    train_src: str,
    train_tgt: str,
    val_src: str,
    val_tgt: str,
    src_lang: str = "de",
    tgt_lang: str = "en",
    batch_size: int = 32,
    min_freq: int = 2,
    max_len: int = 256,
    vocab_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Create training and validation dataloaders.

    Args:
        train_src: Path to training source file
        train_tgt: Path to training target file
        val_src: Path to validation source file
        val_tgt: Path to validation target file
        src_lang: Source language code
        tgt_lang: Target language code
        batch_size: Batch size
        min_freq: Minimum word frequency for vocabulary
        max_len: Maximum sequence length
        vocab_dir: Directory to save/load vocabulary

    Returns:
        Tuple of (train_loader, val_loader, src_vocab, tgt_vocab)
    """
    # Get tokenizers
    src_tokenizer = get_tokenizer(src_lang)
    tgt_tokenizer = get_tokenizer(tgt_lang)

    # Build or load vocabularies
    src_vocab_path = (
        os.path.join(vocab_dir, f"src_vocab_{src_lang}.pkl") if vocab_dir else None
    )
    tgt_vocab_path = (
        os.path.join(vocab_dir, f"tgt_vocab_{tgt_lang}.pkl") if vocab_dir else None
    )

    if vocab_dir and os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        print("Loading existing vocabularies...")
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        print("Building vocabularies...")
        src_vocab = build_vocabulary(train_src, src_tokenizer, min_freq)
        tgt_vocab = build_vocabulary(train_tgt, tgt_tokenizer, min_freq)

        if vocab_dir:
            os.makedirs(vocab_dir, exist_ok=True)
            src_vocab.save(src_vocab_path)
            tgt_vocab.save(tgt_vocab_path)

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    # Create datasets
    train_dataset = TranslationDataset(
        train_src,
        train_tgt,
        src_vocab,
        tgt_vocab,
        src_tokenizer,
        tgt_tokenizer,
        max_len,
    )
    val_dataset = TranslationDataset(
        val_src, val_tgt, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    pad_idx = src_vocab.pad_idx

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, src_vocab, tgt_vocab


# =============================================================================
# Synthetic Data for Testing
# =============================================================================


def create_synthetic_data(
    vocab_size: int = 100,
    num_samples: int = 1000,
    seq_len: int = 20,
    batch_size: int = 32,
) -> Tuple[DataLoader, Vocabulary, Vocabulary]:
    """
    Create synthetic copy task data for testing.
    The model learns to copy input sequence to output.
    """
    vocab = Vocabulary()
    for i in range(vocab_size - 4):  # Subtract special tokens
        vocab.add_token(str(i))

    # Generate random sequences
    data = []
    for _ in range(num_samples):
        length = torch.randint(5, seq_len, (1,)).item()
        seq = torch.randint(4, vocab_size, (length,))  # Avoid special tokens
        # Add SOS/EOS
        src = torch.cat(
            [torch.tensor([vocab.sos_idx]), seq, torch.tensor([vocab.eos_idx])]
        )
        tgt = src.clone()
        data.append((src, tgt))

    # Create dataloader
    def collate(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(
            src_batch, batch_first=True, padding_value=vocab.pad_idx
        )
        tgt_padded = pad_sequence(
            tgt_batch, batch_first=True, padding_value=vocab.pad_idx
        )
        return Batch(src_padded, tgt_padded, vocab.pad_idx)

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate)

    return loader, vocab, vocab


if __name__ == "__main__":
    # Test synthetic data creation
    print("Testing data module with synthetic data...")
    loader, src_vocab, tgt_vocab = create_synthetic_data(
        vocab_size=50, num_samples=100, batch_size=8
    )

    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  src: {batch.src.shape}")
    print(f"  src_mask: {batch.src_mask.shape}")
    print(f"  tgt: {batch.tgt.shape}")
    print(f"  tgt_y: {batch.tgt_y.shape}")
    print(f"  tgt_mask: {batch.tgt_mask.shape}")
    print(f"  ntokens: {batch.ntokens}")

    print("\n✅ Data module test passed!")
