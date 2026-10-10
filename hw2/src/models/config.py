from dataclasses import dataclass

from .feature_extractor import FEATURE_EXTRACTOR_MODES
from .kernels import KernelName


@dataclass(frozen=True)
class MosaicConfig:
    """Гиперпараметры одной сети Memory Mosaic.

    Один объект передаётся стеку, блоку и обеим памятям, чтобы размеры,
    свёртки и dropout не повторялись в каждом конструкторе.
    """

    vocab_size: int
    n_head: int = 12
    n_embd: int = 768
    n_layer: int = 12
    block_size: int = 512
    v_shift: int = 1
    k_fe_type: FEATURE_EXTRACTOR_MODES = "linearconv"
    v_fe_type: FEATURE_EXTRACTOR_MODES = "lowrlinearconv"
    k_kernel_size: int = 1
    v_kernel_size: int = 2
    ic_dropout: float = 0.05
    hd_dropout: float = 0.05
    bias: bool = False
    pmem_count: int = 1
    pmem_size: int = 2688
    weight_tying: bool = True
    skip_tokens: int = 0
    attn_only: bool = False
    kernel: KernelName = "softmax"
    kernel_topk: int = 8

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd % n_head must be 0, got n_embd: {self.n_embd} "
                f"n_head: {self.n_head}"
            )
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be at least 1, got {self.n_layer}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be at least 1, got {self.block_size}")
        if self.kernel_topk < 1:
            raise ValueError(f"kernel_topk must be at least 1, got {self.kernel_topk}")
