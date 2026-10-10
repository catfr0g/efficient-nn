import torch
from torch import nn

from .config import MosaicConfig
from .kernels import attention_weights
from .feature_extractor_factory import factory_feature_extractor
from .leaky_keys import (
    EXP_PARAM_SCALE,
    KERNEL_BETA_UPBOUND,
    VALUE_BETA_INIT,
    apply_leaky_keys,
    attention_mask,
    future_mask,
    lag_coefficients,
    value_scale,
)


class InContextAssoMemBlock(nn.Module):
    """Контекстная память одного блока.

    Ключ собирается из текущего и прошлых входов. Значение смешивает текущий
    вход со следующим. Выход позиции t — среднее значений с индексами строго
    меньше t; правило весов задаёт config.kernel. Первые v_shift строк
    обнуляются: у них нет прошлого.
    """

    def __init__(self, config: MosaicConfig):
        super().__init__()
        self.config = config
        self.att_shift = 0
        self.attn_mask = attention_mask(config.block_size, config.v_shift, self.att_shift)
        self.leaky_attn_mask = future_mask(config.block_size)
        self.k_featurizer = factory_feature_extractor(config.k_fe_type)(
            shift=0,
            norm=False,
            kernel_size=config.k_kernel_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            bias=config.bias,
        )
        self.v_featurizer = factory_feature_extractor(config.v_fe_type)(
            shift=config.v_shift,
            norm=True,
            kernel_size=config.v_kernel_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            bias=config.bias,
        )
        self.ic_dropout = nn.Dropout(config.ic_dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.hd_dropout)
        self.leakyavg_coef = lag_coefficients(config.block_size)
        self.leaky_key_beta = nn.Parameter(
            torch.linspace(0.5, 5, config.n_head) / EXP_PARAM_SCALE
        )
        self.kernel_beta = nn.Parameter(torch.ones(config.n_head) / EXP_PARAM_SCALE)
        self.kernel_beta_upbound = KERNEL_BETA_UPBOUND
        self.value_beta = nn.Parameter(
            torch.ones(config.n_head) * VALUE_BETA_INIT / EXP_PARAM_SCALE
        )
        self.device: torch.device | str = "cpu"

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
            scale_power=1,
        )
        values = self.v_featurizer(x) * value_scale(
            self.value_beta, n_head, length, head_dim
        ).expand(batch, n_head, length, head_dim)
        attention = keys @ keys.transpose(-1, -2)
        attention = attention.masked_fill(
            self.attn_mask[:, :, :length, :length], float("-inf")
        )
        start = self.config.v_shift + self.att_shift
        attention = attention_weights(
            attention, self.config.kernel, topk=self.config.kernel_topk
        )
        if start:
            rows = torch.arange(length, device=attention.device).view(1, 1, -1, 1)
            attention = attention.masked_fill(rows < start, 0)
        attention = self.ic_dropout(attention)
        output = attention @ values
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.c_proj(output))
