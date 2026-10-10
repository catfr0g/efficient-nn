"""Урезанный RegBench: несколько строк из одного случайного автомата.

Каждая строка — путь по детерминированному автомату. Из состояния есть
несколько рёбер, следующее слово выбирается равномерно. Строки разделены
SEP. Модель предсказывает следующий символ; сам SEP в лосс не входит.
Accuracy — совпадение последнего символа. TVD — среднее по всем
предсказанным символам расстояние до равномерного распределения на
разрешённых рёбрах.
"""

import random
from dataclasses import dataclass

import torch

ALPHABET = 8
SEP = ALPHABET
PAD = ALPHABET + 1
VOCAB_SIZE = ALPHABET + 2
PAD_TO = 128
MIN_STRINGS = 8
MAX_STRINGS = 12
MIN_LEN = 2
MAX_LEN = 6
MIN_STATES = 4
MAX_STATES = 8
MAX_EDGES = 3


@dataclass(frozen=True)
class Batch:
    idx: torch.Tensor
    targets: torch.Tensor
    legal: torch.Tensor


def _automaton(rng: random.Random) -> list[list[int]]:
    """transitions[state][symbol] — следующее состояние или -1, если ребра нет."""
    n_states = rng.randint(MIN_STATES, MAX_STATES)
    transitions = [[-1] * ALPHABET for _ in range(n_states)]
    for state in range(n_states):
        others = [item for item in range(n_states) if item != state]
        symbols = rng.sample(range(ALPHABET), rng.randint(1, MAX_EDGES))
        for symbol in symbols:
            transitions[state][symbol] = rng.choice(others)
    return transitions


def _walk(transitions: list[list[int]], rng: random.Random, length: int) -> list[int]:
    state = 0
    symbols: list[int] = []
    for _ in range(length):
        legal = [symbol for symbol, nxt in enumerate(transitions[state]) if nxt >= 0]
        choice = rng.choice(legal)
        symbols.append(choice)
        state = transitions[state][choice]
    return symbols


def generate_example(
    rng: random.Random,
    transitions: list[list[int]] | None = None,
) -> tuple[list[int], list[int], list[list[int]]]:
    """Вход, цели и списки разрешённых символов на каждой позиции входа."""
    if transitions is None:
        transitions = _automaton(rng)
    strings = [
        _walk(transitions, rng, rng.randint(MIN_LEN, MAX_LEN))
        for _ in range(rng.randint(MIN_STRINGS, MAX_STRINGS))
    ]
    full: list[int] = []
    for index, symbols in enumerate(strings):
        if index:
            full.append(SEP)
        full.extend(symbols)
    state = transitions[0][full[0]]
    targets: list[int] = []
    allowed: list[list[int]] = []
    for nxt in full[1:]:
        if nxt == SEP:
            targets.append(-1)
            allowed.append([])
            state = 0
            continue
        legal = [symbol for symbol, dst in enumerate(transitions[state]) if dst >= 0]
        if nxt not in legal:
            raise RuntimeError("sampled symbol is not a legal edge")
        targets.append(nxt)
        allowed.append(legal)
        state = transitions[state][nxt]
    return full[:-1], targets, allowed


def collate(
    examples: list[tuple[list[int], list[int], list[list[int]]]],
    device: torch.device,
) -> Batch:
    idx: list[list[int]] = []
    targets: list[list[int]] = []
    legal = torch.zeros(len(examples), PAD_TO, VOCAB_SIZE, device=device)
    for row, (tokens, example_targets, allowed) in enumerate(examples):
        if len(tokens) > PAD_TO:
            raise ValueError(f"example length {len(tokens)} exceeds {PAD_TO}")
        pad = PAD_TO - len(tokens)
        idx.append(tokens + [PAD] * pad)
        targets.append(example_targets + [-1] * pad)
        for position, symbols in enumerate(allowed):
            if not symbols:
                continue
            share = 1.0 / len(symbols)
            for symbol in symbols:
                legal[row, position, symbol] = share
    return Batch(
        idx=torch.tensor(idx, dtype=torch.long, device=device),
        targets=torch.tensor(targets, dtype=torch.long, device=device),
        legal=legal,
    )
