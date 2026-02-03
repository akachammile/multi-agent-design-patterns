"""
Transformer Model Implementation

This is a PyTorch implementation of the Transformer model as described in:
"Attention Is All You Need" (Vaswani et al., 2017)
Paper: https://arxiv.org/abs/1706.03762

Source Reference:
- Harvard NLP "The Annotated Transformer": https://github.com/harvardnlp/annotated-transformer
- Original TensorFlow implementation (Tensor2Tensor): https://github.com/tensorflow/tensor2tensor

Author: Based on Harvard NLP implementation
License: MIT
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Helper Functions
# =============================================================================


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers.

    Args:
        module: The module to clone
        n: Number of clones to produce

    Returns:
        ModuleList containing n copies of the module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (for decoder self-attention).
    Creates a lower triangular matrix of 1s.

    Args:
        size: Size of the square mask

    Returns:
        Tensor of shape (1, size, size) with upper triangle masked out
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# =============================================================================
# Core Components
# =============================================================================


class LayerNorm(nn.Module):
    """
    Layer Normalization as described in the paper.

    Construct a layernorm module.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note: for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer) -> Tensor:
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


# =============================================================================
# Attention Mechanism
# =============================================================================


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute 'Scaled Dot Product Attention'.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        query: Query tensor of shape (batch, heads, seq_len, d_k)
        key: Key tensor of shape (batch, heads, seq_len, d_k)
        value: Value tensor of shape (batch, heads, seq_len, d_v)
        mask: Optional mask tensor
        dropout: Optional dropout layer

    Returns:
        Tuple of (output tensor, attention weights)
    """
    d_k = query.size(-1)
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    p_attn = F.softmax(scores, dim=-1)

    # Apply dropout (if provided)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Compute weighted sum of values
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        Args:
            h: Number of attention heads
            d_model: Model dimension
            dropout: Dropout probability
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # Linear projections for Q, K, V and output
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Implements Multi-Head Attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # Delete query, key, value to free memory
        del query
        del key
        del value

        return self.linears[-1](x)


# =============================================================================
# Feed Forward Network
# =============================================================================


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            dropout: Dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# =============================================================================
# Embeddings
# =============================================================================


class Embeddings(nn.Module):
    """
    Token embeddings with learned weights.
    """

    def __init__(self, d_model: int, vocab: int):
        """
        Args:
            d_model: Model dimension
            vocab: Vocabulary size
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        # Scale embeddings by sqrt(d_model) as in the paper
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# =============================================================================
# Encoder Components
# =============================================================================


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of self-attention and feed-forward network.

    Each layer has two sub-layers:
    1. Multi-head self-attention mechanism
    2. Position-wise fully connected feed-forward network

    Both sub-layers employ a residual connection followed by layer normalization.
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        """
        Args:
            size: Model dimension
            self_attn: Multi-head attention module
            feed_forward: Feed-forward network module
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Apply self-attention then feed-forward with residual connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    """

    def __init__(self, layer: EncoderLayer, N: int):
        """
        Args:
            layer: Encoder layer to stack
            N: Number of layers
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# Decoder Components
# =============================================================================


class DecoderLayer(nn.Module):
    """
    Decoder layer consisting of self-attention, cross-attention, and feed-forward network.

    Each layer has three sub-layers:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention over encoder output
    3. Position-wise fully connected feed-forward network
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        """
        Args:
            size: Model dimension
            self_attn: Self-attention module
            src_attn: Cross-attention module (encoder-decoder attention)
            feed_forward: Feed-forward network module
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor],
        tgt_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Apply self-attention, cross-attention, then feed-forward with residual connections.

        Args:
            x: Target sequence embeddings
            memory: Encoder output
            src_mask: Source mask
            tgt_mask: Target mask (causal)
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer: DecoderLayer, N: int):
        """
        Args:
            layer: Decoder layer to stack
            N: Number of layers
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor],
        tgt_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# =============================================================================
# Generator (Output Layer)
# =============================================================================


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model: int, vocab: int):
        """
        Args:
            d_model: Model dimension
            vocab: Vocabulary size
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


# =============================================================================
# Full Encoder-Decoder Transformer
# =============================================================================


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        """
        Args:
            encoder: Encoder module
            decoder: Decoder module
            src_embed: Source embedding + positional encoding
            tgt_embed: Target embedding + positional encoding
            generator: Output generation module
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor],
        tgt_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: Tensor, src_mask: Optional[Tensor]) -> Tensor:
        """
        Encode the source sequence.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: Tensor,
        src_mask: Optional[Tensor],
        tgt: Tensor,
        tgt_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Decode using encoder memory and target sequence.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# =============================================================================
# Model Factory Function
# =============================================================================


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """
    Helper function to construct a Transformer model from hyperparameters.

    Default hyperparameters correspond to the "base" model in the paper.

    Args:
        src_vocab: Source vocabulary size
        tgt_vocab: Target vocabulary size
        N: Number of encoder and decoder layers (default: 6)
        d_model: Model dimension (default: 512)
        d_ff: Feed-forward dimension (default: 2048)
        h: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)

    Returns:
        Transformer model ready for training
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# =============================================================================
# Test and Demo
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check
    print("=" * 60)
    print("Transformer Model Test")
    print("=" * 60)

    # Create a small model for testing
    src_vocab = 1000
    tgt_vocab = 1000
    model = make_model(src_vocab, tgt_vocab, N=2, d_model=256, d_ff=512, h=4)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Create dummy input
    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    # Create masks
    src_mask = torch.ones(batch_size, 1, src_len)
    tgt_mask = subsequent_mask(tgt_len)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)
        logits = model.generator(output)

    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    print(f"\nOutput shapes:")
    print(f"  Decoder output: {output.shape}")
    print(f"  Logits: {logits.shape}")

    print("\n✅ Transformer model test passed!")
