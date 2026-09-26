"""Уравнения производительности для первого задания."""

from dataclasses import dataclass

import numpy as np


type NumberOrArray = int | float | np.ndarray

WEIGHT_BYTES = 4_159_872


@dataclass(frozen=True)
class LatencyTheta:
    t0: float
    flop_rate: float
    byte_rate: float


@dataclass(frozen=True)
class EnergyTheta:
    p0: float
    e_flop: float
    e_byte: float


def flops(image_size: NumberOrArray, batch: NumberOrArray) -> NumberOrArray:
    """Число FLOPs за прямой проход."""
    return batch * (17_714 * image_size**2 + 313_344)


def bytes_moved(image_size: NumberOrArray, batch: NumberOrArray) -> NumberOrArray:
    """Оценка объёма перемещённых данных в байтах."""
    return batch * (196 * image_size**2 + 8_592) + WEIGHT_BYTES


def memory(image_size: NumberOrArray, batch: NumberOrArray) -> NumberOrArray:
    """Оценка пикового объёма памяти в байтах."""
    return 52 * batch * image_size**2 + WEIGHT_BYTES


def latency(
    image_size: NumberOrArray, batch: NumberOrArray, theta: LatencyTheta
) -> NumberOrArray:
    """Оценка задержки в секундах."""
    return theta.t0 + np.maximum(
        flops(image_size, batch) / theta.flop_rate,
        bytes_moved(image_size, batch) / theta.byte_rate,
    )


def energy(
    image_size: NumberOrArray,
    batch: NumberOrArray,
    theta: LatencyTheta,
    theta_energy: EnergyTheta,
) -> NumberOrArray:
    """Оценка энергии в джоулях."""
    return (
        theta_energy.p0 * latency(image_size, batch, theta)
        + theta_energy.e_flop * flops(image_size, batch)
        + theta_energy.e_byte * bytes_moved(image_size, batch)
    )
