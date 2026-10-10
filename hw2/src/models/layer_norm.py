import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm с необязательным bias. У nn.LayerNorm его нельзя просто выключить."""

    def __init__(self, ndim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self._eps)
