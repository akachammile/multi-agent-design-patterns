import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in the Transformer paper.
    
    It consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: The number of expected features in the input (usually 512).
            d_ff: The dimension of the feedforward layer (usually 2048).
            dropout: The dropout probability.
        """
        super(FeedForward, self).__init__()
        # First linear layer: projects from d_model to d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # Second linear layer: projects from d_ff back to d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply first linear layer followed by ReLU activation
        x = F.relu(self.w_1(x))
        # Apply dropout
        x = self.dropout(x)
        # Apply second linear layer
        x = self.w_2(x)
        return x

if __name__ == "__main__":
    # Quick test
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10
    
    fnn = FeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    output = fnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert x.shape == output.shape
    print("FNN test passed!")
