import torch
from torch import nn

from .config import MosaicConfig
from .in_context_asso_mem_block import InContextAssoMemBlock
from .layer_norm import LayerNorm
from .persistent_asso_mem_block import PersistentAssoMemBlock


class MosaicBlock(nn.Module):
    """Один блок Memory Mosaic с pre-LN остатками.

    Сначала контекстная память: x + InContext(LayerNorm(x)).
    Если attn_only выключен, следом слоты: x + Persistent(LayerNorm(x)).
    """

    def __init__(self, config: MosaicConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = InContextAssoMemBlock(config)
        self.ln2: LayerNorm | None
        self.pmem: PersistentAssoMemBlock | None
        if config.attn_only:
            self.ln2 = None
            self.pmem = None
        else:
            self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
            self.pmem = PersistentAssoMemBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.pmem is not None and self.ln2 is not None:
            x = x + self.pmem(self.ln2(x))
        return x
