"""Reverse и sort: exact match по длине.

Оба задания делят словарь и раскладку примера. Reverse в статье — 4 слоя
и тест до 1.5× (96). Sort — 2 слоя и тест до 4× (256). Бюджет здесь общий
с уже посчитанным reverse: ширина 256, 512 слотов, батч 64, 8000 шагов.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .algo import SOURCE_LEN, VOCAB_SIZE, Batch, collate, generate_example, sequence_length
from .models import KernelName, MosaicConfig, StackAssoMem
from .train import KERNELS, _bind_train_step, forward_loss

STEPS = 8_000
BATCH_SIZE = 64
EVAL_EXAMPLES = 256
OVERFIT_LOSS = 0.05
OVERFIT_STEPS = 2_000
LOG_EVERY = 10
EVAL_EVERY = 2_000
TRAIN_PAD = sequence_length(SOURCE_LEN)


@dataclass(frozen=True)
class Task:
    name: str
    n_layer: int
    block_size: int
    eval_lengths: tuple[int, ...]
    protocol: str


TASKS = {
    "reverse": Task(
        name="reverse",
        n_layer=4,
        block_size=256,
        eval_lengths=(64, 96),
        protocol=(
            "Reverse vs paper Table 2: 4 layers vs 4, width 256 vs 512, "
            "8 heads vs 8, persistent slots 512 vs 2688, batch 64 vs 256, "
            "8000 steps (512k examples) vs 30M examples, test 64 and 96, one seed. "
            "Train content length is 64."
        ),
    ),
    "sort": Task(
        name="sort",
        n_layer=2,
        block_size=1024,
        eval_lengths=(64, 128, 256),
        protocol=(
            "Sort vs paper Table 2: 2 layers vs 2, width 256 vs 512, "
            "8 heads vs 8, persistent slots 512 vs 2688, batch 64 vs 256, "
            "8000 steps (512k examples) vs 40M examples, test 64, 128 and 256, "
            "one seed. Train content length is 64."
        ),
    ),
}


def make_model(kernel: KernelName, device: torch.device, task: Task) -> StackAssoMem:
    torch.manual_seed(0)
    config = MosaicConfig(
        vocab_size=VOCAB_SIZE,
        n_head=8,
        n_embd=256,
        n_layer=task.n_layer,
        block_size=task.block_size,
        pmem_size=512,
        ic_dropout=0.0,
        hd_dropout=0.0,
        kernel=kernel,
        kernel_topk=8,
    )
    model = StackAssoMem(config)
    model.to(device)
    return model


def optimizer(model: StackAssoMem, device: torch.device) -> torch.optim.Optimizer:
    return model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), device.type)


def make_batch(
    task: Task,
    step: int,
    length: int,
    count: int,
    device: torch.device,
    pad_to: int | None,
) -> Batch:
    rng = random.Random(step)
    examples = [generate_example(task.name, length, rng) for _ in range(count)]
    return collate(examples, device, pad_to=pad_to)


@torch.no_grad()
def exact_match(model: StackAssoMem, batch: Batch) -> tuple[int, int]:
    was_training = model.training
    model.eval()
    logits = model(batch.idx, logits_only=True)
    if was_training:
        model.train()
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits_only did not return logits")
    predicted = logits.argmax(dim=-1).gather(1, batch.logit_positions)
    gold = batch.targets.gather(1, batch.logit_positions)
    matches = (predicted == gold).all(dim=-1)
    return int(matches.sum()), int(matches.numel())


def exact_match_length(task: Task, model: StackAssoMem, length: int, device: torch.device) -> float:
    if length >= 256:
        chunk = 4
    elif length > SOURCE_LEN:
        chunk = 8
    else:
        chunk = 16
    correct = 0
    total = 0
    remaining = EVAL_EXAMPLES
    offset = 50_000 + length
    while remaining:
        count = min(chunk, remaining)
        batch = make_batch(task, offset + total, length, count, device, pad_to=None)
        matched, size = exact_match(model, batch)
        correct += matched
        total += size
        remaining -= count
    return correct / total


def run_overfit(
    task: Task,
    kernel: KernelName,
    batch: Batch,
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[bool, float, int]:
    model = make_model(kernel, device, task)
    opt = optimizer(model, device)
    model.train()
    loss_value = 0.0
    progress = tqdm(range(1, OVERFIT_STEPS + 1), desc=f"overfit {kernel}")
    for step in progress:
        loss = forward_loss(model, batch.idx, batch.targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())
        progress.set_postfix(loss=f"{loss_value:.3f}")
        if step % LOG_EVERY == 0 or loss_value < OVERFIT_LOSS:
            writer.add_scalar("overfit/loss", loss_value, step)
        if loss_value < OVERFIT_LOSS:
            return True, loss_value, step
    return False, loss_value, OVERFIT_STEPS


def run_train(task: Task, kernel: KernelName, device: torch.device, writer: SummaryWriter) -> StackAssoMem:
    model = make_model(kernel, device, task)
    opt = optimizer(model, device)
    model.train()
    train_step = _bind_train_step(model, opt)
    if next(model.parameters()).is_cuda:
        print(f"compiling {kernel}", flush=True)
    progress = tqdm(range(1, STEPS + 1), desc=f"train {kernel}")
    for step in progress:
        batch = make_batch(task, step, SOURCE_LEN, BATCH_SIZE, device, pad_to=TRAIN_PAD)
        loss = train_step(batch.idx, batch.targets)
        loss_value = float(loss)
        progress.set_postfix(loss=f"{loss_value:.3f}")
        if step % LOG_EVERY == 0:
            writer.add_scalar("train/loss", loss_value, step)
        if step % EVAL_EVERY == 0:
            score = exact_match_length(task, model, SOURCE_LEN, device)
            writer.add_scalar("eval/exact_match_64", score, step)
            model.train()
    return model


def write_table(path: Path, rows: list[dict[str, str]], lengths: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kernel", *map(str, lengths)])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), default="reverse")
    args = parser.parse_args()
    task = TASKS[args.task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(task.protocol, flush=True)
    print(f"device {device}", flush=True)
    log_root = Path("hw2/results/tb") / task.name
    table_path = Path("hw2/results") / f"{task.name}_table.csv"
    held_out = make_batch(task, 0, SOURCE_LEN, 32, device, pad_to=TRAIN_PAD)
    passed: list[KernelName] = []
    for kernel in KERNELS:
        writer = SummaryWriter(log_dir=log_root / kernel)
        ok, loss_value, taken = run_overfit(task, kernel, held_out, device, writer)
        print(
            f"overfit {kernel} done loss {loss_value:.4f} steps {taken} ok {ok}",
            flush=True,
        )
        writer.close()
        if ok:
            passed.append(kernel)
    rows: list[dict[str, str]] = []
    for kernel in KERNELS:
        if kernel not in passed:
            rows.append(
                {
                    "kernel": kernel,
                    **{str(length): "overfit_failed" for length in task.eval_lengths},
                }
            )
            continue
        writer = SummaryWriter(log_dir=log_root / kernel)
        model = run_train(task, kernel, device, writer)
        row = {"kernel": kernel}
        for length in task.eval_lengths:
            score = exact_match_length(task, model, length, device)
            row[str(length)] = f"{score:.4f}"
            writer.add_scalar("eval/exact_match", score, length)
            print(f"eval {kernel} length {length} exact {score:.4f}", flush=True)
        writer.close()
        rows.append(row)
        write_table(table_path, rows, task.eval_lengths)


if __name__ == "__main__":
    main()
