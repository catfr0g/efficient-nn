"""Подбор параметров аналитической модели по измерениям."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .equations import EnergyTheta, LatencyTheta, bytes_moved, energy, flops, latency, memory


RESULTS = Path(__file__).resolve().parent / "results"


def _prepare(data: pd.DataFrame) -> pd.DataFrame:
    required = {"S", "B", "latency", "energy", "is_validation"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"В measurements.csv отсутствуют столбцы: {', '.join(sorted(missing))}")

    rows = data.copy()
    flags = rows["is_validation"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if flags.isna().any():
        raise ValueError("is_validation должен содержать true/false или 1/0")
    rows["is_validation"] = flags.astype(bool)

    for column in ("S", "B", "latency", "energy"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    sizes = rows["S"].to_numpy(dtype=float)
    batches = rows["B"].to_numpy(dtype=float)
    if not np.all(np.isfinite(sizes) & (sizes >= 32) & (sizes % 16 == 0)):
        raise ValueError("S должен быть кратен 16 и не меньше 32")
    if not np.all(np.isfinite(batches) & (batches > 0) & (batches % 1 == 0)):
        raise ValueError("B должен быть положительным целым числом")

    rows["_oom"] = False
    if "memory" in rows:
        rows["_oom"] = rows["memory"].astype(str).str.strip().str.upper().eq("OOM")
        rows["memory"] = pd.to_numeric(rows["memory"], errors="coerce")
    return rows


def _fit_latency(rows: pd.DataFrame) -> LatencyTheta:
    rows = rows.loc[rows["latency"].gt(0) & np.isfinite(rows["latency"])]
    if len(rows) < 3:
        raise ValueError("Для подбора задержки нужны хотя бы три измерения")

    sizes = rows["S"].to_numpy(dtype=float)
    batches = rows["B"].to_numpy(dtype=float)
    measured = rows["latency"].to_numpy(dtype=float)
    operations = np.asarray(flops(sizes, batches), dtype=float)
    traffic = np.asarray(bytes_moved(sizes, batches), dtype=float)
    f_scale, q_scale, t_scale = operations.max(), traffic.max(), measured.max()
    f, q, target = operations / f_scale, traffic / q_scale, measured / t_scale

    def residual(parameters: np.ndarray) -> np.ndarray:
        t0, compute, transfer = parameters
        return t0 + np.maximum(compute * f, transfer * q) - target

    initial = (
        (target.min() / 2, 1.0, 1.0),
        (target.min() / 2, 1.0, 0.1),
        (target.min() / 2, 0.1, 1.0),
    )
    fits = [
        least_squares(
            residual,
            guess,
            bounds=(0, np.inf),
            loss="soft_l1",
            f_scale=0.03,
            max_nfev=10_000,
        )
        for guess in initial
    ]
    successful = [fit for fit in fits if fit.success]
    if not successful:
        raise ValueError("Не удалось подобрать параметры задержки")
    fit = min(successful, key=lambda result: result.cost)
    t0, compute, transfer = fit.x
    if compute <= 1e-9 or transfer <= 1e-9:
        raise ValueError("Для оценки обеих скоростей нужны точки в обоих режимах")
    return LatencyTheta(
        t0=float(t0 * t_scale),
        flop_rate=float(f_scale / (compute * t_scale)),
        byte_rate=float(q_scale / (transfer * t_scale)),
    )


def _fit_energy(rows: pd.DataFrame, theta: LatencyTheta) -> EnergyTheta:
    rows = rows.loc[rows["energy"].gt(0) & np.isfinite(rows["energy"])]
    if len(rows) < 3:
        raise ValueError("Для подбора энергии нужны хотя бы три измерения")

    sizes = rows["S"].to_numpy(dtype=float)
    batches = rows["B"].to_numpy(dtype=float)
    measured = rows["energy"].to_numpy(dtype=float)
    features = np.column_stack(
        (latency(sizes, batches, theta), flops(sizes, batches), bytes_moved(sizes, batches))
    )
    scales = features.max(axis=0)
    normalized = features / scales
    if np.linalg.matrix_rank(normalized) < 3:
        raise ValueError("Для подбора энергии нужны измерения с разными S и B")

    coefficients, _ = nnls(normalized / measured[:, None], np.ones_like(measured))
    p0, e_flop, e_byte = coefficients / scales
    return EnergyTheta(float(p0), float(e_flop), float(e_byte))


def calibrate(data: pd.DataFrame) -> tuple[LatencyTheta, EnergyTheta]:
    """Подбирает параметры только по тренировочным измерениям без OOM."""
    rows = _prepare(data)
    train = rows.loc[~rows["is_validation"] & ~rows["_oom"]]
    theta = _fit_latency(train)
    return theta, _fit_energy(train, theta)


def _report(rows: pd.DataFrame, theta: LatencyTheta, theta_energy: EnergyTheta) -> None:
    for is_validation, label in ((False, "Обучение"), (True, "Проверка")):
        part = rows.loc[rows["is_validation"].eq(is_validation) & ~rows["_oom"]]
        for column, predict in (
            ("latency", lambda s, b: latency(s, b, theta)),
            ("energy", lambda s, b: energy(s, b, theta, theta_energy)),
        ):
            valid = part.loc[part[column].gt(0) & np.isfinite(part[column])]
            if valid.empty:
                continue
            measured = valid[column].to_numpy(dtype=float)
            predicted = predict(valid["S"].to_numpy(), valid["B"].to_numpy())
            median_error = np.median(np.abs(predicted - measured) / measured)
            print(f"{label}, {column}: {len(valid)} точек, медианная ошибка {median_error:.1%}")


def _plot(
    rows: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    scale: float,
    predict: Callable[[np.ndarray, int], np.ndarray],
    output: Path,
) -> None:
    batches = sorted(rows["B"].unique())
    columns = min(3, len(batches))
    nrows = math.ceil(len(batches) / columns)
    figure, axes = plt.subplots(
        nrows, columns,
        figsize=(4.5 * columns, 2.9 * nrows),
        sharex=True, layout="constrained",
    )
    axes = np.asarray(axes).reshape(-1)
    max_size = int(rows["S"].max())
    sizes = np.arange(32, max_size + 1, 16)

    for batch, axis in zip(batches, axes):
        part = rows.loc[rows["B"].eq(batch) & ~rows["_oom"]
                        & rows[column].gt(0) & np.isfinite(rows[column])]
        axis.plot(sizes, predict(sizes, int(batch)) / scale, color="#222222", label="Модель")
        for is_validation, label, color in (
            (False, "Обучение", "#1f77b4"),
            (True, "Проверка", "#d95f02"),
        ):
            points = part.loc[part["is_validation"].eq(is_validation)]
            axis.scatter(points["S"], points[column] / scale, s=25, color=color,
                         zorder=3, label=label)
        if column == "memory":
            oom = rows.loc[rows["B"].eq(batch) & rows["_oom"]]
            axis.scatter(oom["S"], predict(oom["S"].to_numpy(), int(batch)) / scale,
                         marker="x", s=50, color="#c23b22", zorder=4,
                         label="OOM (y: прогноз)")
        axis.set(title=f"B = {int(batch)}", yscale="log", xlim=(24, max_size * 1.02))
        axis.set_xticks([size for size in (32, 128, 256, 384, 512, 1024) if size <= max_size])
        axis.grid(alpha=0.25, which="both")

    for axis in axes[len(batches):]:
        axis.set_visible(False)
    axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.supxlabel("Размер изображения S, пиксели")
    figure.supylabel(ylabel)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plots(rows: pd.DataFrame, theta: LatencyTheta,
           theta_energy: EnergyTheta, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for column, title, ylabel, scale, predict in (
        ("latency", "Задержка: измерения и модель", "Задержка одного прохода, мс", 1e-3,
         lambda s, b: latency(s, b, theta)),
        ("memory", "Пиковая память: измерения и модель", "Пиковая память, МиБ", 2**20,
         memory),
        ("energy", "Энергия: измерения и модель", "Энергия одного прохода, Дж", 1,
         lambda s, b: energy(s, b, theta, theta_energy)),
    ):
        path = directory / f"{column}.png"
        _plot(rows, column, title, ylabel, scale, predict, path)
        print(f"График сохранён в {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=RESULTS / "measurements.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "theta.json")
    parser.add_argument("--figures", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    if not args.measurements.is_file():
        parser.error(f"Файл измерений не найден: {args.measurements}")

    data = pd.read_csv(args.measurements)
    theta, theta_energy = calibrate(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"latency": asdict(theta), "energy": asdict(theta_energy)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    rows = _prepare(data)
    _report(rows, theta, theta_energy)
    _plots(rows, theta, theta_energy, args.figures)
    print(f"Параметры сохранены в {args.output}")


if __name__ == "__main__":
    main()
