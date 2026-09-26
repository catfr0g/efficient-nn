"""Измерения времени, памяти, энергии и CUDA-ядер на сетке задания."""

import argparse
import csv
import gc
import random
import statistics
import time
from pathlib import Path
from typing import Iterable

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from .models import HomeworkCNN


RESULTS = Path(__file__).resolve().parent / "results"
BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MEASUREMENT_FIELDS = ("S", "B", "latency", "memory", "energy", "is_validation")
KERNEL_FIELDS = ("S", "B", "layer", "kernel name")


def grid(seed: int) -> list[tuple[int, int, bool]]:
    rng = random.Random(seed)
    extra_sizes = rng.sample([s for s in range(32, 513, 16) if s not in BASE_SIZES], 4)
    extra_batches = rng.sample([b for b in range(1, 257) if b not in BASE_BATCHES], 3)
    pairs = [(s, b) for s in sorted((*BASE_SIZES, *extra_sizes))
             for b in sorted((*BASE_BATCHES, *extra_batches))]
    rng.shuffle(pairs)
    return [(s, b, i % 5 == 0) for i, (s, b) in enumerate(pairs)]


def _rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(fields):
            raise ValueError(f"Неверные столбцы в {path}: {reader.fieldnames}")
        return list(reader)


def _append(path: Path, fields: tuple[str, ...],
            rows: Iterable[dict[str, object]] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def _measure(model: HomeworkCNN, image_size: int, batch: int,
             energy_device: object, warmup: int, min_seconds: float) -> tuple[float, int, float, torch.Tensor]:
    x = torch.randn(batch, 3, image_size, image_size, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        output = model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del output

        samples = []
        energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(energy_device)
        start = time.perf_counter()
        while time.perf_counter() - start < min_seconds or len(samples) < 5:
            before = time.perf_counter()
            output = model(x)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - before)
            del output
        energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(energy_device)

    return statistics.median(samples), peak, (energy_end - energy_start) / (1000 * len(samples)), x


def _kernels(model: HomeworkCNN, x: torch.Tensor) -> list[tuple[str, str]]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with torch.inference_mode():
            for index, layer in enumerate(model):
                with record_function(f"layer_{index:02d}_{type(layer).__name__}"):
                    x = layer(x)
        torch.cuda.synchronize()

    events = profiler.events()
    operations = {event.id: event for event in events
                  if event.device_type == torch.autograd.DeviceType.CPU
                  and event.name.startswith("aten::")}
    result = []
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA or not event.linked_correlation_id:
            continue
        parent = operations.get(event.linked_correlation_id)
        while parent is not None and not parent.name.startswith("layer_"):
            parent = parent.cpu_parent
        if parent is not None:
            result.append((parent.name, event.name))
    if not result:
        raise RuntimeError("Профилировщик не вернул CUDA-ядра")
    return result


def _measure_point(model: HomeworkCNN, point: tuple[int, int, bool],
                   energy_device: object, args: argparse.Namespace,
                   progress: str) -> torch.Tensor | None:
    image_size, batch, is_validation = point
    row: dict[str, object] = {"S": image_size, "B": batch, "is_validation": is_validation}
    try:
        latency, memory, energy, x = _measure(
            model, image_size, batch, energy_device, args.warmup, args.min_seconds
        )
    except torch.cuda.OutOfMemoryError:
        row.update(latency="", memory="OOM", energy="")
        _append(args.measurements, MEASUREMENT_FIELDS, [row])
        print(f"{progress}: OOM", flush=True)
        return None

    row.update(latency=latency, memory=memory, energy=energy)
    _append(args.measurements, MEASUREMENT_FIELDS, [row])
    print(f"{progress}: {latency:.6f} s, {memory / 2**20:.1f} MiB, {energy:.6f} J", flush=True)
    return x


def _profile_point(model: HomeworkCNN, x: torch.Tensor | None,
                   point: tuple[int, int, bool], path: Path, progress: str) -> None:
    image_size, batch, _ = point
    if x is None:
        x = torch.randn(batch, 3, image_size, image_size, device="cuda")
    try:
        kernels = _kernels(model, x)
    except torch.cuda.OutOfMemoryError:
        kernels = [("profiling", "OOM")]
        print(f"{progress}: OOM при профилировании", flush=True)
    rows = ({"S": image_size, "B": batch, "layer": layer, "kernel name": name}
            for layer, name in kernels)
    _append(path, KERNEL_FIELDS, rows)


def _run_grid(args: argparse.Namespace, model: HomeworkCNN, energy_device: object) -> None:
    measurements = _rows(args.measurements, MEASUREMENT_FIELDS)
    done = {(int(row["S"]), int(row["B"])) for row in measurements}
    oom = {(int(row["S"]), int(row["B"])) for row in measurements if row["memory"] == "OOM"}
    profiled = {(int(row["S"]), int(row["B"]))
                for row in _rows(args.kernels, KERNEL_FIELDS)}
    points = grid(args.seed)[:args.limit]
    _append(args.measurements, MEASUREMENT_FIELDS)
    _append(args.kernels, KERNEL_FIELDS)

    for index, point in enumerate(points, 1):
        image_size, batch, _ = point
        key = image_size, batch
        if key in oom or (key in done and key in profiled):
            continue
        progress = f"{index}/{len(points)} S={image_size} B={batch}"
        x = None
        try:
            if key not in done:
                x = _measure_point(model, point, energy_device, args, progress)
                if x is None:
                    continue
            if key not in profiled:
                _profile_point(model, x, point, args.kernels, progress)
        finally:
            del x
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=RESULTS / "measurements.csv")
    parser.add_argument("--kernels", type=Path, default=RESULTS / "kernels.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Число конфигураций для пробного запуска")
    parser.add_argument("--memory-limit-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.warmup < 0 or args.min_seconds <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("warmup >= 0, min-seconds > 0 и limit >= 1")
    if not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if not 0 < args.memory_limit_gib * 2**30 <= total_memory:
        parser.error("memory-limit-gib должен быть положительным и не больше VRAM")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 2**30 / total_memory)
    print(f"Лимит аллокатора: {args.memory_limit_gib:g} ГиБ", flush=True)
    model = HomeworkCNN().cuda().eval()

    pynvml.nvmlInit()
    try:
        uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
        energy_device = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        _run_grid(args, model, energy_device)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
