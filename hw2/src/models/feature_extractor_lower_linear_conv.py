import torch
from torch import nn

from .feature_extractor import FeatureExtractor


class FeatureExtractorLowerLinearConv(FeatureExtractor):
    """Линейная проекция и смесь текущего входа со сдвинутым на shift."""
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
        if kernel_size != 2:
            raise ValueError(
                "fe_type == lowrlinearconv only support kernel_size=2, "
                f"got kernel_size: {kernel_size}"
            )
        self.shiftpad = nn.ZeroPad1d((-shift, shift))
        self.linear = nn.Linear(n_embd, n_embd, bias=bias)
        self.coef = nn.Parameter(torch.ones(n_head).uniform_(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _channels = x.size()
        head_dim = self.n_embd // self.n_head
        x = self.linear(x).transpose(1, 2).view(batch, self.n_embd, length)
        current = x.view(batch, self.n_head, head_dim, length).transpose(2, 3)
        shifted = self.shiftpad(x).view(batch, self.n_head, head_dim, length)
        shifted = shifted.transpose(2, 3)
        coef = self.coef.view(1, self.n_head, 1, 1)
        y = shifted * (1 - coef) + current * coef
        if self.norm:
            y = y / y.norm(dim=-1, keepdim=True)
        return y
