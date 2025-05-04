from torch import nn


def medusa_layer(medusa_num_heads, hidden_size, vocab_size):
    """
    Constructs Medusa layer
    Args:
        medusa_num_heads: Number of medusa head
        hidden_size: Size of the hidden layers in the block
        vocab_size: Size of the vocabulary
    """
    return nn.ModuleList(
        [
            nn.Sequential(
                ResidualBlock(hidden_size),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(medusa_num_heads)
        ]
    )

class ResidualBlock(nn.Module):
    """
    A Residual Block module: performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection
    Args:
        hidden_size: Size of the hidden layers in the block
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResidualBlock
        Args:
            x: Input tensor
        Returns:
            Output after the residual connection
        """
        return x + self.act(self.linear(x))
    