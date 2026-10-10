import torch
from torch import nn

from .feature_extractor import FeatureExtractor


class FeatureExtractorLinearConv(FeatureExtractor):
    """Свёртка по каналам. При ядре длины 1 соседние позиции не смешиваются."""
    def __init__(
        self,
        shift: int,
        norm: bool,
        kernel_size: int,
        n_embd: int,
        n_head: int,
        bias: bool,
    ):
        super().__init__(shift, norm, kernel_size, n_embd, n_head, bias)
        self.shiftpad = nn.ZeroPad1d((kernel_size - 1 - shift, shift))
        self.conv = nn.Conv1d(
            in_channels=n_embd, out_channels=n_embd, kernel_size=kernel_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _channels = x.size()
        head_dim = self.n_embd // self.n_head
        x = x.transpose(1, 2)
        y = self.conv(self.shiftpad(x)).view(batch, self.n_head, head_dim, length)
        y = y.transpose(2, 3)
        if self.norm:
            y = y / y.norm(dim=-1, keepdim=True)
        return y
