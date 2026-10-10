import inspect
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import MosaicConfig
from .layer_norm import LayerNorm
from .mosaic_block import MosaicBlock


class StackAssoMem(nn.Module):
    """Сеть Memory Mosaic: эмбеддинг, стопка блоков и слой в словарь.

    Эмбеддинг и выходной слой делят одну матрицу, если weight_tying включён.
    Логиты делятся на √d. Позиционного эмбеддинга нет.
    """

    def __init__(self, config: MosaicConfig):
        super().__init__()
        self.config = config
        self.eval_status: list[float] = []
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[MosaicBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_out = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.emb.weight = self.head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                fanin = param.shape[1]
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=1 / math.sqrt(fanin) / math.sqrt(config.n_layer),
                )
        self.device: torch.device | str = "cpu"

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        logits_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        x = self.ln_out(self.blocks(self.emb(idx)))
        logits_scale = 1.0 / math.sqrt(self.config.n_embd)
        if logits_only:
            return logits_scale * self.head(x[:, self.config.skip_tokens :])
        if targets is not None:
            logits = logits_scale * self.head(x[:, self.config.skip_tokens :])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets[:, self.config.skip_tokens :].contiguous().view(-1),
                ignore_index=-1,
            )
            return logits, loss
        logits = logits_scale * self.head(x[:, [-1], :])
        return logits, None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            fanin = module.weight.shape[1]
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / np.sqrt(fanin))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            fanin = module.weight.shape[1] * module.weight.shape[2]
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / np.sqrt(fanin))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1)

    def to(self, *args: Any, **kwargs: Any) -> "StackAssoMem":
        super().to(*args, **kwargs)
        non_blocking = kwargs.get("non_blocking", False)
        if len(args) >= 2 and isinstance(args[1], bool):
            non_blocking = args[1]
        elif len(args) >= 3 and isinstance(args[2], bool):
            non_blocking = args[2]
        if not isinstance(non_blocking, bool):
            non_blocking = False
        device = self.emb.weight.device
        first = self.blocks[0]
        if not isinstance(first, MosaicBlock):
            raise ValueError(f"Found not MosaicBlock[0]: {first}")
        leaky_attn_mask = first.attn.leaky_attn_mask.to(
            device, non_blocking=non_blocking
        )
        attn_mask = first.attn.attn_mask.to(device, non_blocking=non_blocking)
        leakyavg_coef = first.attn.leakyavg_coef.to(device, non_blocking=non_blocking)
        for index, block in enumerate(self.blocks):
            if not isinstance(block, MosaicBlock):
                raise ValueError(f"Found not MosaicBlock[{index}]: {block}")
            block.attn.leaky_attn_mask = leaky_attn_mask
            block.attn.attn_mask = attn_mask
            block.attn.leakyavg_coef = leakyavg_coef
            block.attn.device = device
            if block.pmem is not None:
                block.pmem.leakyavg_coef = leakyavg_coef
                block.pmem.leaky_attn_mask = leaky_attn_mask
        self.device = device
        return self

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        decay_params: list[nn.Parameter] = []
        nodecay_params: list[nn.Parameter] = []
        for name, param in param_dict.items():
            if "time_" in name or "gate" in name or param.dim() < 2:
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(param.numel() for param in decay_params)
        num_nodecay_params = sum(param.numel() for param in nodecay_params)
        print(
            "num decayed parameter tensors: "
            f"{len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            "num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args: dict[str, bool] = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

    def checkparams(self) -> dict[str, object]:
        return {}

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        storyid: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token: int | None = None,
        skip_future: bool = True,
        future_topk: int = 2,
    ) -> tuple[torch.Tensor, None]:
        del storyid, skip_future, future_topk
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("generate expects logits from forward")
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if stop_token is not None and int(idx_next) == stop_token:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, None

    @torch.no_grad()
    def beam_generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        beam_width: int,
        temperature: float = 1.0,
        stop_token: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        if idx.size(0) != 1:
            raise ValueError(f"idx.size(0) must be 1, got {idx.size(0)}")
        bank: list[tuple[torch.Tensor, float, int]] = []
        final_bank: list[tuple[torch.Tensor, float, int]] = []
        topkprob, topkidx = self._next_topk(idx, beam_width, temperature)
        for i, token in enumerate(topkidx[0]):
            if stop_token is not None and int(token) == stop_token:
                continue
            bank.append(
                (torch.cat([idx, token.view(1, 1)], dim=1), topkprob[0][i].item(), 1)
            )
        while len(bank) > 0:
            temp_bank: list[tuple[torch.Tensor, float, int]] = []
            for sequence, prob, length in bank:
                if length >= max_new_tokens:
                    final_bank.append((sequence, prob, length))
                    continue
                topkprob, topkidx = self._next_topk(sequence, beam_width, temperature)
                for i, token in enumerate(topkidx[0]):
                    if stop_token is not None and int(token) == stop_token:
                        final_bank.append((sequence, prob, length))
                        continue
                    temp_bank.append(
                        (
                            torch.cat([sequence, token.view(1, 1)], dim=1),
                            topkprob[0][i].item() * prob,
                            length + 1,
                        )
                    )
            if len(temp_bank) == 0:
                break
            bank = temp_bank
            order = np.argsort([item[1] for item in bank])[-beam_width:]
            bank = [bank[int(position)] for position in order]
        if len(final_bank) == 0:
            raise RuntimeError("beam_generate produced no finished sequences")
        best = max(range(len(final_bank)), key=lambda position: final_bank[position][1])
        return final_bank[best][0], None

    def _next_topk(
        self, idx: torch.Tensor, beam_width: int, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_cond = (
            idx
            if idx.size(1) <= self.config.block_size
            else idx[:, -self.config.block_size :]
        )
        logits, _ = self(idx_cond)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("beam_generate expects logits from forward")
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.topk(probs, k=beam_width, dim=1)
