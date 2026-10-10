from typing import Literal

from torch import nn

FEATURE_EXTRACTOR_MODES = Literal["linearconv", "lowrlinearconv"]


class FeatureExtractor(nn.Module):
    """Проекция входа в представление одной головы: ключ или значение."""
    def __init__(
        self,
        shift: int,
        norm: bool,
        kernel_size: int,
        n_embd: int,
        n_head: int,
        bias: bool,
    ):
        super().__init__()
        self.norm = norm
        self.n_head = n_head
        self.n_embd = n_embd
