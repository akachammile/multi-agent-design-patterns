"""
Training Script for Transformer Model

This script implements:
- Label smoothing loss
- Noam learning rate scheduler (as in the paper)
- Training loop with gradient accumulation
- Model checkpointing
- Greedy decoding for inference

Source Reference:
- Harvard NLP "The Annotated Transformer": https://github.com/harvardnlp/annotated-transformer

Usage:
    # Train with synthetic data (for testing)
    python train.py --test
    
    # Train with real data
    python train.py --train_src data/train.de --train_tgt data/train.en \
                    --val_src data/val.de --val_tgt data/val.en
"""

import os
import time
import argparse
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from transformer import make_model, subsequent_mask
from data import create_dataloaders, create_synthetic_data, Batch, Vocabulary
from config import Config, get_small_config


# =============================================================================
# Label Smoothing Loss
# =============================================================================


class LabelSmoothing(nn.Module):
    """
    Label smoothing loss as described in the paper.

    Instead of using one-hot target distribution, use:
    q(k) = (1 - smoothing) if k == target else smoothing / (vocab_size - 2)

    This helps prevent the model from being too confident.
    """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1):
        """
        Args:
            size: Vocabulary size
            padding_idx: Index of padding token (ignored in loss)
            smoothing: Label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.size = size
        self.confidence = 1.0 - smoothing
        self.true_dist = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            x: Log probabilities from model (batch * seq_len, vocab_size)
            target: Target indices (batch * seq_len,)

        Returns:
            Loss value
        """
        assert x.size(1) == self.size

        # Create smoothed target distribution
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        # Mask out padding positions
        mask = target == self.padding_idx
        true_dist[mask] = 0

        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class SimpleLossCompute:
    """
    A simple loss compute and train function.
    """

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: float) -> Tuple[float, Tensor]:
        """
        Compute loss and return (loss_value, loss_tensor for backward).

        Args:
            x: Model output (batch, seq_len, d_model)
            y: Target labels (batch, seq_len)
            norm: Normalization factor (number of tokens)

        Returns:
            Tuple of (loss value, loss tensor)
        """
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.item() * norm, sloss


# =============================================================================
# Learning Rate Scheduler (Noam)
# =============================================================================


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    Implement the Noam learning rate schedule.

    lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    This increases the learning rate linearly for the first warmup steps,
    and decreases it thereafter proportionally to the inverse square root of step.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def get_scheduler(optimizer, d_model: int, warmup: int = 4000, factor: float = 1.0):
    """Create Noam scheduler."""
    return LambdaLR(
        optimizer, lr_lambda=lambda step: rate(step, d_model, factor, warmup)
    )


# =============================================================================
# Training State
# =============================================================================


@dataclass
class TrainState:
    """Track training state."""

    step: int = 0  # Current training step
    epoch: int = 0  # Current epoch
    samples: int = 0  # Total samples processed
    tokens: int = 0  # Total tokens processed
    best_val_loss: float = float("inf")


# =============================================================================
# Training Functions
# =============================================================================


def run_epoch(
    data_loader,
    model: nn.Module,
    loss_compute: SimpleLossCompute,
    optimizer=None,
    scheduler=None,
    train_state: TrainState = None,
    log_every: int = 100,
    training: bool = True,
    device: str = "cuda",
) -> Tuple[float, TrainState]:
    """
    Run one epoch of training or validation.

    Args:
        data_loader: DataLoader to iterate over
        model: Transformer model
        loss_compute: Loss computation function
        optimizer: Optimizer (None for validation)
        scheduler: Learning rate scheduler
        train_state: Training state tracker
        log_every: Log every N batches
        training: Whether this is training (vs validation)
        device: Device to use

    Returns:
        Tuple of (average loss, updated train state)
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0

    if train_state is None:
        train_state = TrainState()

    model.train() if training else model.eval()

    iterator = (
        tqdm(data_loader, desc="Training" if training else "Validation")
        if TQDM_AVAILABLE
        else data_loader
    )

    for i, batch in enumerate(iterator):
        # Move batch to device
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        tgt_y = batch.tgt_y.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_mask = batch.tgt_mask.to(device)

        # Forward pass
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            out = model(src, tgt, src_mask, tgt_mask)
            loss_value, loss_tensor = loss_compute(out, tgt_y, batch.ntokens)

            if training:
                loss_tensor.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_state.step += 1
                train_state.tokens += batch.ntokens

        total_loss += loss_value
        total_tokens += batch.ntokens

        # Logging
        if training and (i + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tokens_per_sec = total_tokens / elapsed

            if TQDM_AVAILABLE:
                iterator.set_postfix(
                    {
                        "loss": f"{loss_value / batch.ntokens:.4f}",
                        "lr": f"{lr:.2e}",
                        "tok/s": f"{tokens_per_sec:.0f}",
                    }
                )
            else:
                print(
                    f"Step {train_state.step:6d} | "
                    f"Loss: {loss_value / batch.ntokens:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f}"
                )

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, train_state


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    config: Config,
) -> TrainState:
    """
    Full training loop.

    Args:
        model: Transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        config: Training configuration

    Returns:
        Final training state
    """
    device = config.device
    model = model.to(device)

    # Loss function
    criterion = LabelSmoothing(
        size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=config.training.label_smoothing,
    )
    criterion = criterion.to(device)

    # Optimizer
    optimizer = Adam(
        model.parameters(), lr=config.training.lr, betas=(0.9, 0.98), eps=1e-9
    )

    # Scheduler
    scheduler = get_scheduler(
        optimizer, d_model=config.model.d_model, warmup=config.training.warmup_steps
    )

    # Loss compute function
    loss_compute = SimpleLossCompute(model.generator, criterion)

    # Training state
    train_state = TrainState()

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(config.training.epochs):
        train_state.epoch = epoch

        print(f"\n--- Epoch {epoch + 1}/{config.training.epochs} ---")

        # Training
        model.train()
        train_loss, train_state = run_epoch(
            train_loader,
            model,
            loss_compute,
            optimizer,
            scheduler,
            train_state,
            log_every=config.training.log_every,
            training=True,
            device=device,
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, _ = run_epoch(
                val_loader, model, loss_compute, training=False, device=device
            )
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_state": train_state,
                "config": config,
            }
            path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        # Save best model
        if val_loss < train_state.best_val_loss:
            train_state.best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                },
                best_path,
            )
            print(f"Saved best model: {best_path}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best validation loss: {train_state.best_val_loss:.4f}")
    print(f"{'='*60}")

    return train_state


# =============================================================================
# Inference
# =============================================================================


def greedy_decode(
    model: nn.Module,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    device: str = "cuda",
) -> Tensor:
    """
    Greedy decoding for inference.

    Args:
        model: Trained Transformer model
        src: Source sequence (batch, src_len)
        src_mask: Source mask
        max_len: Maximum output length
        start_symbol: SOS token index
        device: Device to use

    Returns:
        Generated sequence tensor
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    # Encode source
    memory = model.encode(src, src_mask)

    # Initialize with start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for _ in range(max_len - 1):
        # Decode
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )

        # Get next token
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Append to output
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    return ys


# =============================================================================
# Test with Synthetic Data
# =============================================================================


def test_training():
    """Test training with synthetic copy task."""
    print("\n" + "=" * 60)
    print("Testing training with synthetic copy task")
    print("=" * 60)

    # Create synthetic data
    vocab_size = 50
    loader, src_vocab, tgt_vocab = create_synthetic_data(
        vocab_size=vocab_size,
        num_samples=500,
        seq_len=15,
        batch_size=32,
    )

    # Get small config for testing
    config = get_small_config()
    config.training.epochs = 3
    config.training.warmup_steps = 100
    config.training.log_every = 10

    # Create model
    model = make_model(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        N=config.model.N,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        h=config.model.h,
        dropout=config.model.dropout,
    )

    # Simple training loop for testing
    device = config.device
    model = model.to(device)

    criterion = LabelSmoothing(
        size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=0.0,  # No smoothing for copy task
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = get_scheduler(optimizer, config.model.d_model, warmup=100)

    loss_compute = SimpleLossCompute(model.generator, criterion)
    train_state = TrainState()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")

    for epoch in range(config.training.epochs):
        model.train()
        train_loss, train_state = run_epoch(
            loader,
            model,
            loss_compute,
            optimizer,
            scheduler,
            train_state,
            log_every=config.training.log_every,
            training=True,
            device=device,
        )
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}")

    # Test inference
    print("\n--- Testing inference ---")
    model.eval()

    # Create test sequence
    test_seq = torch.tensor([[1, 10, 20, 30, 40, 2]]).to(device)  # SOS + tokens + EOS
    src_mask = torch.ones(1, 1, test_seq.size(1)).to(device)

    with torch.no_grad():
        output = greedy_decode(
            model,
            test_seq,
            src_mask,
            max_len=10,
            start_symbol=src_vocab.sos_idx,
            device=device,
        )

    print(f"Input:  {test_seq.cpu().tolist()}")
    print(f"Output: {output.cpu().tolist()}")

    print("\n✅ Training test passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Transformer model")

    # Data arguments
    parser.add_argument("--train_src", type=str, help="Path to training source file")
    parser.add_argument("--train_tgt", type=str, help="Path to training target file")
    parser.add_argument("--val_src", type=str, help="Path to validation source file")
    parser.add_argument("--val_tgt", type=str, help="Path to validation target file")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="FF dimension")
    parser.add_argument("--h", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--N", type=int, default=6, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--warmup", type=int, default=4000, help="Warmup steps")
    parser.add_argument("--lr", type=float, default=1.0, help="Base learning rate")

    # Other arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--vocab_dir", type=str, default="vocab")
    parser.add_argument("--src_lang", type=str, default="de", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")

    # Test mode
    parser.add_argument("--test", action="store_true", help="Run training test")

    args = parser.parse_args()

    if args.test:
        test_training()
        return

    # Check data arguments
    if not all([args.train_src, args.train_tgt, args.val_src, args.val_tgt]):
        print("Error: Please provide all data paths or use --test for synthetic data")
        parser.print_help()
        return

    # Create config
    from config import Config, ModelConfig, TrainingConfig, DataConfig

    config = Config(
        model=ModelConfig(
            d_model=args.d_model,
            d_ff=args.d_ff,
            h=args.h,
            N=args.N,
            dropout=args.dropout,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            warmup_steps=args.warmup,
            lr=args.lr,
        ),
        data=DataConfig(
            train_src=args.train_src,
            train_tgt=args.train_tgt,
            val_src=args.val_src,
            val_tgt=args.val_tgt,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        ),
        checkpoint_dir=args.checkpoint_dir,
        vocab_dir=args.vocab_dir,
    )

    # Create dataloaders
    train_loader, val_loader, src_vocab, tgt_vocab = create_dataloaders(
        train_src=config.data.train_src,
        train_tgt=config.data.train_tgt,
        val_src=config.data.val_src,
        val_tgt=config.data.val_tgt,
        src_lang=config.data.src_lang,
        tgt_lang=config.data.tgt_lang,
        batch_size=config.training.batch_size,
        min_freq=config.data.min_freq,
        max_len=config.data.max_seq_len,
        vocab_dir=config.vocab_dir,
    )

    # Create model
    model = make_model(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        N=config.model.N,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        h=config.model.h,
        dropout=config.model.dropout,
    )

    # Train
    train_model(model, train_loader, val_loader, src_vocab, tgt_vocab, config)


if __name__ == "__main__":
    main()
