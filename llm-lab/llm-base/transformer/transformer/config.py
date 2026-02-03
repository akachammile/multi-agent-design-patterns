"""
Training Configuration for Transformer Model

This module contains all configuration parameters for training.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Transformer model configuration."""

    d_model: int = 512  # Model dimension
    d_ff: int = 2048  # Feed-forward dimension
    h: int = 8  # Number of attention heads
    N: int = 6  # Number of encoder/decoder layers
    dropout: float = 0.1  # Dropout probability
    max_len: int = 5000  # Maximum sequence length for positional encoding


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    epochs: int = 10
    lr: float = 1.0  # Base learning rate (scaled by Noam scheduler)
    warmup_steps: int = 4000  # Warmup steps for learning rate scheduling
    label_smoothing: float = 0.1
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Checkpointing
    save_every: int = 1  # Save checkpoint every N epochs
    log_every: int = 100  # Log every N batches

    # Mixed precision
    use_amp: bool = True  # Use automatic mixed precision


@dataclass
class DataConfig:
    """Data configuration."""

    # Data paths
    train_src: str = "data/train.de"
    train_tgt: str = "data/train.en"
    val_src: str = "data/val.de"
    val_tgt: str = "data/val.en"

    # Tokenizer settings
    src_lang: str = "de"  # Source language
    tgt_lang: str = "en"  # Target language
    min_freq: int = 2  # Minimum word frequency for vocabulary
    max_seq_len: int = 256  # Maximum sequence length

    # Special tokens
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    checkpoint_dir: str = "checkpoints"
    vocab_dir: str = "vocab"

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        model = ModelConfig(**config_dict.get("model", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        data = DataConfig(**config_dict.get("data", {}))

        return cls(
            model=model,
            training=training,
            data=data,
            device=config_dict.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            ),
            checkpoint_dir=config_dict.get("checkpoint_dir", "checkpoints"),
            vocab_dir=config_dict.get("vocab_dir", "vocab"),
            seed=config_dict.get("seed", 42),
        )


# Default configuration instances
def get_base_config() -> Config:
    """Get base model configuration (as in the paper)."""
    return Config()


def get_small_config() -> Config:
    """Get small model configuration for testing/debugging."""
    return Config(
        model=ModelConfig(d_model=256, d_ff=512, h=4, N=2),
        training=TrainingConfig(batch_size=16, epochs=5, warmup_steps=1000),
    )


if __name__ == "__main__":
    config = get_base_config()
    print("Base Configuration:")
    print(
        f"  Model: d_model={config.model.d_model}, h={config.model.h}, N={config.model.N}"
    )
    print(
        f"  Training: batch_size={config.training.batch_size}, epochs={config.training.epochs}"
    )
    print(f"  Device: {config.device}")
