"""Reverse и sort: цепочка чисел, затем тот же ряд задом наперёд или по возрастанию.

Формат совпадает с синтетикой статьи: [BOS] исходная цепочка [SEP] ответ.
Словарь — числа 0..31 и четыре служебных токена, всего 36. Токен EOS в ответ
не входит: exact match смотрит только на переставленные числа.
"""

import random
from dataclasses import dataclass

import torch

CONTENT = 32
BOS = 32
SEP = 33
EOS = 34
PAD = 35
VOCAB_SIZE = 36
SOURCE_LEN = 64


@dataclass(frozen=True)
class Batch:
    idx: torch.Tensor
    targets: torch.Tensor
    logit_positions: torch.Tensor


def generate_example(
    kind: str, length: int, rng: random.Random
) -> tuple[list[int], list[int], list[int]]:
    """Один пример. length — сколько чисел в исходной цепочке.

    targets[t] равен ответу, который должна предсказать позиция t.
    На исходной цепочке и на EOS стоит -1.
    """
    if kind not in ("reverse", "sort"):
        raise ValueError(f"unknown algo task: {kind}")
    if length < 1:
        raise ValueError(f"length must be at least 1, got {length}")
    source = [rng.randrange(CONTENT) for _ in range(length)]
    answer = list(reversed(source)) if kind == "reverse" else sorted(source)
    sequence = [BOS, *source, SEP, *answer, EOS]
    tokens = sequence[:-1]
    targets = [-1] * len(tokens)
    first = length + 1
    positions = list(range(first, first + length))
    for offset, position in enumerate(positions):
        targets[position] = answer[offset]
    return tokens, targets, positions


def sequence_length(length: int) -> int:
    """Длина входа: BOS, исходная цепочка, SEP и ответ, без EOS."""
    return 2 * length + 2


def collate(
    examples: list[tuple[list[int], list[int], list[int]]],
    device: torch.device,
    pad_to: int | None = None,
) -> Batch:
    width = max(len(tokens) for tokens, _, _ in examples)
    if pad_to is not None:
        if width > pad_to:
            raise ValueError(f"example length {width} exceeds pad_to {pad_to}")
        width = pad_to
    idx: list[list[int]] = []
    targets: list[list[int]] = []
    positions: list[list[int]] = []
    for tokens, example_targets, example_positions in examples:
        pad = width - len(tokens)
        idx.append(tokens + [PAD] * pad)
        targets.append(example_targets + [-1] * pad)
        positions.append(example_positions)
    return Batch(
        idx=torch.tensor(idx, dtype=torch.long, device=device),
        targets=torch.tensor(targets, dtype=torch.long, device=device),
        logit_positions=torch.tensor(positions, dtype=torch.long, device=device),
    )
