import math

import torch
from torch import nn
from torch.nn import functional as F


class Pmem(nn.Module):
    """Softmax-внимание запроса к фиксированным слотам ключ-значение.

    Слоты хранятся как параметры модуля, форма (1, головы, слоты, dim).
    Батч подхватывается обычным умножением матриц, без expand: у
    scaled_dot_product_attention растянутый по батчу тензор на CUDA давал
    illegal memory access.
    """

    def __init__(
        self,
        pmem_count: int,
        pmem_size: int,
        pmem_dim: int,
        pmem_nhead: int,
        dropout: float,
    ):
        super().__init__()
        if pmem_count < 1:
            raise ValueError(f"pmem_count must be at least 1, got {pmem_count}")
        self.pmem_size = pmem_size
        self.pmem_dim = pmem_dim
        self.pmem_nhead = pmem_nhead
        self.pmem_count = pmem_count
        self.dropout = dropout
        self.M_k = nn.ParameterList(
            self._new_slot(pmem_nhead, pmem_size, pmem_dim) for _ in range(pmem_count)
        )
        self.M_v = nn.ParameterList(
            self._new_slot(pmem_nhead, pmem_size, pmem_dim) for _ in range(pmem_count)
        )

    @staticmethod
    def _new_slot(n_head: int, pmem_size: int, pmem_dim: int) -> nn.Parameter:
        weight = torch.empty(1, n_head, pmem_size, pmem_dim)
        nn.init.normal_(weight, mean=0.0, std=1 / math.sqrt(pmem_dim))
        return nn.Parameter(weight)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        total = self._attend(key, self.M_k[0], self.M_v[0])
        for index in range(1, self.pmem_count):
            total = total + self._attend(key, self.M_k[index], self.M_v[index])
        return total / self.pmem_count

    def _attend(
        self, query: torch.Tensor, slot_k: torch.Tensor, slot_v: torch.Tensor
    ) -> torch.Tensor:
        scores = query @ slot_k.transpose(-1, -2)
        weights = torch.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout)
        return weights @ slot_v
