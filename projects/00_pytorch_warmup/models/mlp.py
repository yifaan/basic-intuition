"""TODO: Implement a simple MLP for Lab 0.

Goals:
- Build a manual Linear layer (weight + bias + forward).
- Stack a small MLP with ReLU activations.
- Flatten input inside the forward pass.
"""

import torch
from torch import nn

# TODO: implement LinearManual
# 1) class LinearManual(nn.Module)
# 2) __init__(self, in_features, out_features):
#    - create weight (out_features, in_features)
#    - create bias (out_features,)
#    - wrap with nn.Parameter
# 3) forward(self, x): return x @ weight.T + bias
class LinearManual(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

# TODO: implement MLP
# 1) class MLP(nn.Module)
# 2) __init__(self, in_dim=28*28, num_classes=10):
#    - create 2–3 LinearManual layers
#    - add nn.ReLU() activations
# 3) forward(self, x):
#    - flatten: x = x.view(x.size(0), -1)
#    - apply linear + relu stacks
#    - return logits
class MLP(nn.Module):
    def __init__(self, in_dim=28*28, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            LinearManual(in_dim, 128),
            nn.ReLU(),
            LinearManual(128, 64),
            nn.ReLU(),
            LinearManual(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Optional sanity check (after implementation):
# - model = MLP(); x = torch.randn(4, 1, 28, 28)
# - assert model(x).shape == (4, 10)

if __name__ == "__main__":
    # Sanity check
    model = MLP()
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
    print("MLP implementation is correct.")
