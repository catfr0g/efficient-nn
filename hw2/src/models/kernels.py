from typing import Literal

import torch
import triton
import triton.language as tl

KernelName = Literal[
    "softmax",
    "sparsemax",
    "norm_relu",
    "entmax15",
    "entmax43",
    "topk",
]


def attention_weights(
    scores: torch.Tensor,
    kernel: KernelName,
    *,
    topk: int = 8,
) -> torch.Tensor:
    """Веса внимания по строкам скоров.

    Нечисловые позиции (маска −∞) в ядро не входят и получают вес 0.
    Строка без единой живой позиции остаётся нулевой.
    """
    if kernel == "softmax":
        weights = torch.softmax(scores, dim=-1)
        return torch.nan_to_num(weights, nan=0.0)
    if kernel == "sparsemax":
        return _sparsemax(scores)
    if kernel == "norm_relu":
        return _norm_relu(scores)
    if kernel == "entmax15":
        return _entmax(scores, 1.5)
    if kernel == "entmax43":
        return _entmax(scores, 4.0 / 3.0)
    if kernel == "topk":
        return _topk_softmax(scores, topk)
    raise ValueError(f"unknown kernel: {kernel}")


def _sparsemax(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    safe = torch.where(finite, scores, torch.full_like(scores, float("-inf")))
    ordered, _ = torch.sort(safe, dim=-1, descending=True)
    cumulative = ordered.cumsum(dim=-1)
    count = scores.size(-1)
    ranks = torch.arange(1, count + 1, device=scores.device, dtype=scores.dtype)
    ranks = ranks.view(*([1] * (scores.dim() - 1)), count)
    supported = (1 + ranks * ordered > cumulative) & torch.isfinite(ordered)
    support_size = supported.sum(dim=-1, keepdim=True)
    index = (support_size - 1).clamp(min=0)
    tau = (cumulative.gather(-1, index) - 1) / support_size.clamp(min=1)
    weights = torch.where(
        support_size > 0,
        (scores - tau).clamp(min=0),
        torch.zeros_like(scores),
    )
    return weights.masked_fill(~finite, 0)


def _norm_relu(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    positive = torch.where(finite, scores, torch.zeros_like(scores)).clamp(min=0)
    positive = positive * finite
    total = positive.sum(dim=-1, keepdim=True)
    visible = finite.sum(dim=-1, keepdim=True).clamp(min=1)
    uniform = finite.to(dtype=scores.dtype) / visible
    weights = torch.where(total > 0, positive / total.clamp(min=1e-12), uniform)
    return weights * finite


@triton.jit
def _entmax_kernel(
    scores_ptr,
    out_ptr,
    n_cols,
    stride,
    scale,
    BLOCK: tl.constexpr,
    POWER: tl.constexpr,
    ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    scores = tl.load(scores_ptr + row * stride + cols, mask=valid, other=-1.0e30)
    finite = valid & (scores > -1.0e20)
    scaled = tl.where(finite, scale * scores, 0.0)
    highest = tl.max(tl.where(finite, scaled, -1.0e30))
    spread = tl.max(tl.where(finite, tl.abs(scaled), 0.0))
    lower = highest - spread - 10.0  # pyright: ignore[reportOperatorIssue]
    any_finite = tl.max(tl.where(finite, 1.0, 0.0)) > 0.0  # pyright: ignore[reportOperatorIssue]
    lower = tl.where(any_finite, lower, highest)
    upper = highest
    for _ in range(ITERS):
        midpoint = (lower + upper) * 0.5
        piece = tl.maximum(scaled - midpoint, 0.0)
        piece = piece * piece if POWER == 2 else piece * piece * piece
        mass = tl.sum(tl.where(finite, piece, 0.0))
        too_heavy = mass > 1.0  # pyright: ignore[reportOperatorIssue]
        lower = tl.where(too_heavy, midpoint, lower)
        upper = tl.where(too_heavy, upper, midpoint)
    tau = (lower + upper) * 0.5
    piece = tl.maximum(scaled - tau, 0.0)
    piece = piece * piece if POWER == 2 else piece * piece * piece
    tl.store(out_ptr + row * stride + cols, tl.where(finite, piece, 0.0), mask=valid)


class _Entmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, alpha: float) -> torch.Tensor:
        weights = _entmax_forward(scores, alpha)
        ctx.save_for_backward(weights)
        ctx.alpha = alpha
        return weights

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_output = grad_outputs[0]
        (weights,) = ctx.saved_tensors
        scale = ctx.alpha - 1.0
        power = 1.0 / scale
        support = weights > 0
        activated = torch.where(support, weights**scale, torch.zeros_like(weights))
        slope = torch.where(
            support, activated ** (power - 1.0), torch.zeros_like(weights)
        )
        denom = slope.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adjustment = (slope * grad_output).sum(dim=-1, keepdim=True) / denom
        grad_scores = slope * (grad_output - adjustment)
        return grad_scores * support, None


def _entmax_forward(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Triton в обычном прогоне, бисекция на PyTorch под torch.compile.

    Triton-ядро в граф не встраивается, и шаг с ним остаётся почти таким же
    долгим, как без компиляции. Та же бисекция из обычных операций сжимается
    вместе с остальным шагом. 32 итерации совпадают с Triton до ~1e-6.
    """
    if torch.compiler.is_compiling() or not scores.is_cuda or scores.size(-1) > 1024:
        return _entmax_reference(scores, alpha)
    flat = scores.contiguous().reshape(-1, scores.size(-1))
    output = torch.empty_like(flat)
    power = 2 if abs(alpha - 1.5) < 1e-6 else 3
    block = max(triton.next_power_of_2(flat.size(-1)), 16)
    _entmax_kernel[(flat.size(0),)](
        flat,
        output,
        flat.size(-1),
        flat.stride(0),
        alpha - 1.0,
        BLOCK=block,  # pyright: ignore[reportArgumentType]
        POWER=power,  # pyright: ignore[reportArgumentType]
        ITERS=32,  # pyright: ignore[reportArgumentType]
    )
    return output.view_as(scores)


def _entmax(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    return _Entmax.apply(scores, alpha)


def _entmax_reference(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    finite = torch.isfinite(scores)
    scale = alpha - 1.0
    power = 1.0 / scale
    scaled = scale * torch.where(finite, scores, torch.zeros_like(scores))
    highest = scaled.masked_fill(~finite, float("-inf")).amax(dim=-1, keepdim=True)
    spread = scaled.masked_fill(~finite, 0).abs().amax(dim=-1, keepdim=True)
    lower = highest - spread - 10
    upper = highest
    has_row = finite.any(dim=-1, keepdim=True)
    lower = torch.where(has_row, lower, upper)
    for _ in range(32):
        midpoint = (lower + upper) / 2
        mass = ((scaled - midpoint).clamp(min=0) ** power * finite).sum(
            dim=-1, keepdim=True
        )
        too_heavy = mass > 1
        lower = torch.where(too_heavy, midpoint, lower)
        upper = torch.where(too_heavy, upper, midpoint)
    tau = (lower + upper) / 2
    return (scaled - tau).clamp(min=0) ** power * finite


def _topk_softmax(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if topk < 1:
        raise ValueError(f"topk must be at least 1, got {topk}")
    finite = torch.isfinite(scores)
    filled = scores.masked_fill(~finite, float("-inf"))
    k = min(topk, scores.size(-1))
    chosen, index = torch.topk(filled, k, dim=-1)
    selected = torch.nan_to_num(torch.softmax(chosen, dim=-1), nan=0.0)
    weights = torch.zeros_like(scores)
    weights.scatter_(-1, index, selected)
    return weights * finite
