import torch
from torch import nn

from .config import MosaicConfig
from .feature_extractor_factory import factory_feature_extractor
from .leaky_keys import (
    EXP_PARAM_SCALE,
    VALUE_BETA_INIT,
    apply_leaky_keys,
    future_mask,
    lag_coefficients,
    value_scale,
)
from .pmem import Pmem


class PersistentAssoMemBlock(nn.Module):
    """Долговременная память: те же ключи, softmax по обучаемым слотам.

    Слоты не зависят от текущего текста и заменяют feed-forward. Масштаб
    ключа здесь exp(β)²: тот же β задаёт ширину ядра слотов.
    """

    def __init__(self, config: MosaicConfig):
        super().__init__()
        self.config = config
        self.leaky_attn_mask = future_mask(config.block_size)
        self.pmem = Pmem(
            config.pmem_count,
            config.pmem_size,
            config.n_embd // config.n_head,
            config.n_head,
            config.ic_dropout,
        )
        self.k_featurizer = factory_feature_extractor(config.k_fe_type)(
            shift=0,
            norm=False,
            kernel_size=config.k_kernel_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            bias=config.bias,
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.hd_dropout)
        self.leakyavg_coef = lag_coefficients(config.block_size)
        self.leaky_key_beta = nn.Parameter(
            torch.linspace(0.5, 5, config.n_head) / EXP_PARAM_SCALE
        )
        self.kernel_beta = nn.Parameter(torch.ones(config.n_head) / EXP_PARAM_SCALE)
        self.value_beta = nn.Parameter(
            torch.ones(config.n_head) * VALUE_BETA_INIT / EXP_PARAM_SCALE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.size()
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head
        keys = apply_leaky_keys(
            self.k_featurizer(x),
            self.leakyavg_coef,
            self.leaky_attn_mask,
            self.leaky_key_beta,
            self.kernel_beta,
            scale_power=2,
        )
        output = self.pmem(keys) * value_scale(
            self.value_beta, n_head, length, head_dim
        ).expand(batch, n_head, length, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.c_proj(output))
