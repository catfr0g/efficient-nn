"""Упрощённый MQMTAR: несколько пар ключ-значение и четыре запроса.

Числа 0..255 — содержимое пар. Служебные токены идут следом, PAD замыкает
словарь: в длину 259 он не помещается, потому что 0..255 уже заняты.
"""

import random
from dataclasses import dataclass

import torch

CONTENT = 256
PAIR = 256
ASK = 257
ANS = 258
PAD = 259
VOCAB_SIZE = 260

N_QUERIES = 4
PAIR_LEN = 5
TAIL = N_QUERIES * 3 + 1 + 8
MIN_LENGTH = N_QUERIES * PAIR_LEN + TAIL


@dataclass(frozen=True)
class Batch:
    idx: torch.Tensor
    targets: torch.Tensor
    logit_positions: torch.Tensor


def pairs_for_length(length: int) -> int:
    if length < MIN_LENGTH:
        raise ValueError(f"length must be at least {MIN_LENGTH}, got {length}")
    return (length - TAIL) // PAIR_LEN


def generate_example(
    length: int, rng: random.Random
) -> tuple[list[int], list[int], list[int]]:
    """Один пример длины length.

    targets[t] равен idx[t + 1], только когда этот токен входит в восемь
    чисел ответа. Иначе -1. logit_positions — восемь позиций логитов,
    которые эти ответы предсказывают.
    """
    n_pairs = pairs_for_length(length)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    tokens: list[int] = []
    for _ in range(n_pairs):
        key = (rng.randrange(CONTENT), rng.randrange(CONTENT))
        value = (rng.randrange(CONTENT), rng.randrange(CONTENT))
        pairs.append((key, value))
        tokens.extend((PAIR, key[0], key[1], value[0], value[1]))
    chosen = rng.sample(range(n_pairs), N_QUERIES)
    answers: list[int] = []
    for index in chosen:
        key, value = pairs[index]
        tokens.extend((ASK, key[0], key[1]))
        answers.extend(value)
    tokens.append(ANS)
    answer_start = len(tokens)
    tokens.extend(answers)
    if len(tokens) > length:
        raise ValueError(f"example of length {length} built {len(tokens)} tokens")
    tokens.extend([PAD] * (length - len(tokens)))
    targets = [-1] * length
    logit_positions: list[int] = []
    for offset in range(8):
        answer_index = answer_start + offset
        predict_at = answer_index - 1
        targets[predict_at] = tokens[answer_index]
        logit_positions.append(predict_at)
    return tokens, targets, logit_positions


def collate(
    examples: list[tuple[list[int], list[int], list[int]]],
    device: torch.device,
    pad_to: int | None = None,
) -> Batch:
    """Складывает примеры в батч. pad_to задаёт общую длину, если она нужна фиксированной."""
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
