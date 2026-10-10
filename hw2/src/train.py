"""Обучение MQMTAR: tqdm, TensorBoard и таблица exact match по длине."""

import argparse
import csv
import random
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import KernelName, MosaicConfig, StackAssoMem
from .mqmtar import MIN_LENGTH, VOCAB_SIZE, Batch, collate, generate_example

KERNELS: tuple[KernelName, ...] = (
    "softmax",
    "sparsemax",
    "norm_relu",
    "topk",
    "entmax15",
    "entmax43",
)
SMALL_LENGTHS = (64, 128, 256, 512)
SCALED_LENGTHS = (64, 128, 256, 512, 1024)
OVERFIT_LOSS = 0.05
OVERFIT_STEPS = 2000
SMALL_STEPS = 30_000
SCALED_STEPS = 20_000
BUDGET_STEPS = 40_000
BUDGET_BATCH = 256
EVAL_EXAMPLES = 256
LOG_EVERY = 10
EVAL_EVERY = 5_000
TRAIN_LENGTH = 64

PROTOCOL = (
    "Scaled MQMTAR vs paper Table 6: 4 layers vs 4, width 256 vs 512, "
    "8 heads vs 8, persistent slots 512 vs 2688, batch 32 vs 256, "
    "20000 steps (640k examples) vs about 50M examples, test up to 16x "
    "(1024) vs 64x, one seed. Train sequences stay at most 64 tokens."
)
BUDGET_PROTOCOL = (
    "Budget MQMTAR: 4 layers, width 256, 8 heads, persistent slots 512, "
    "batch 256, 40000 steps (10.2M examples), test up to 1024, one seed. "
    "The previous scaled run was batch 32 and 20000 steps (640k examples). "
    "Paper Table 6 is width 512, 2688 slots, batch 256, about 50M examples, "
    "test up to 4096."
)


def small_config(kernel: KernelName) -> MosaicConfig:
    return MosaicConfig(
        vocab_size=VOCAB_SIZE,
        n_head=4,
        n_embd=128,
        n_layer=2,
        block_size=512,
        pmem_size=128,
        ic_dropout=0.0,
        hd_dropout=0.0,
        kernel=kernel,
        kernel_topk=8,
    )


def scaled_config(kernel: KernelName) -> MosaicConfig:
    return MosaicConfig(
        vocab_size=VOCAB_SIZE,
        n_head=8,
        n_embd=256,
        n_layer=4,
        block_size=1024,
        pmem_size=512,
        ic_dropout=0.0,
        hd_dropout=0.0,
        kernel=kernel,
        kernel_topk=8,
    )


def make_model(kernel: KernelName, device: torch.device, scaled: bool) -> StackAssoMem:
    torch.manual_seed(0)
    config = scaled_config(kernel) if scaled else small_config(kernel)
    model = StackAssoMem(config)
    model.to(device)
    return model


def optimizer(model: StackAssoMem, device: torch.device) -> torch.optim.Optimizer:
    return model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), device.type)


def overfit_batch(device: torch.device) -> Batch:
    rng = random.Random(0)
    examples = [generate_example(64, rng) for _ in range(32)]
    return collate(examples, device)


def train_batch(step: int, device: torch.device, batch_size: int) -> Batch:
    """Батч обучения всегда длины 64: короткие примеры дополняются PAD.

    Хвост из PAD не виден позициям ответа из-за каузальной маски, поэтому
    лосс совпадает с батчем без общего паддинга. Фиксированная форма нужна,
    чтобы torch.compile не пересобирал шаг на каждой длине от 41 до 64.
    """
    rng = random.Random(step)
    examples = [
        generate_example(rng.randint(MIN_LENGTH, TRAIN_LENGTH), rng)
        for _ in range(batch_size)
    ]
    return collate(examples, device, pad_to=TRAIN_LENGTH)


def eval_examples(length: int) -> list[tuple[list[int], list[int], list[int]]]:
    rng = random.Random(10_000 + length)
    return [generate_example(length, rng) for _ in range(EVAL_EXAMPLES)]


def forward_loss(
    model: StackAssoMem, idx: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Кросс-энтропия ответов. В обучение передаются уже тензоры, без dataclass."""
    _, loss = model(idx, targets)
    if not isinstance(loss, torch.Tensor):
        raise TypeError("forward did not return a loss")
    return loss


def answer_loss(model: StackAssoMem, batch: Batch) -> torch.Tensor:
    return forward_loss(model, batch.idx, batch.targets)


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


def exact_match_length(model: StackAssoMem, length: int, device: torch.device) -> float:
    chunk = 4 if length >= 512 else 16
    examples = eval_examples(length)
    correct = 0
    total = 0
    for start in range(0, len(examples), chunk):
        matched, count = exact_match(
            model, collate(examples[start : start + chunk], device)
        )
        correct += matched
        total += count
    return correct / total


def run_overfit(
    kernel: KernelName,
    batch: Batch,
    device: torch.device,
    scaled: bool,
    writer: SummaryWriter,
) -> tuple[bool, float, int]:
    model = make_model(kernel, device, scaled)
    opt = optimizer(model, device)
    model.train()
    loss_value = 0.0
    progress = tqdm(range(1, OVERFIT_STEPS + 1), desc=f"overfit {kernel}")
    for step in progress:
        loss = answer_loss(model, batch)
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


def _bind_train_step(
    model: StackAssoMem, opt: torch.optim.Optimizer
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Шаг с фиксированной формой. reduce-overhead держит его в CUDA graph.

    Eval сюда не входит: у него другие длины, и компиляция на них собрала
    бы новый граф. Overfit тоже остаётся обычным циклом: он короче, чем
    сама компиляция.
    """

    def step(idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = forward_loss(model, idx, targets)
        loss.backward()
        opt.step()
        return loss.detach()

    if next(model.parameters()).is_cuda:
        return torch.compile(step, mode="reduce-overhead", dynamic=False)
    return step


def run_train(
    kernel: KernelName,
    device: torch.device,
    scaled: bool,
    steps: int,
    batch_size: int,
    writer: SummaryWriter,
) -> StackAssoMem:
    model = make_model(kernel, device, scaled)
    opt = optimizer(model, device)
    model.train()
    train_step = _bind_train_step(model, opt)
    if next(model.parameters()).is_cuda:
        print(f"compiling {kernel}", flush=True)
    progress = tqdm(range(1, steps + 1), desc=f"train {kernel}")
    for step in progress:
        batch = train_batch(step, device, batch_size)
        loss = train_step(batch.idx, batch.targets)
        loss_value = float(loss)
        progress.set_postfix(loss=f"{loss_value:.3f}")
        if step % LOG_EVERY == 0:
            writer.add_scalar("train/loss", loss_value, step)
        if step % EVAL_EVERY == 0:
            score = exact_match_length(model, 64, device)
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
    parser.add_argument("--overfit", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scaled", action="store_true")
    mode.add_argument("--budget", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaled = bool(args.scaled or args.budget)
    if args.budget:
        lengths = SCALED_LENGTHS
        steps = BUDGET_STEPS
        batch_size = BUDGET_BATCH
        table_name = "length_table_budget.csv"
        log_name = "budget"
    elif args.scaled:
        lengths = SCALED_LENGTHS
        steps = SCALED_STEPS
        batch_size = 32
        table_name = "length_table_scaled.csv"
        log_name = "scaled"
    else:
        lengths = SMALL_LENGTHS
        steps = SMALL_STEPS
        batch_size = 64
        table_name = "length_table.csv"
        log_name = "small"
    table_path = Path("hw2/results") / table_name
    log_root = Path("hw2/results/tb") / log_name
    if args.budget:
        print(BUDGET_PROTOCOL, flush=True)
    elif args.scaled:
        print(PROTOCOL, flush=True)
    print(f"device {device}", flush=True)
    held_out = overfit_batch(device)
    passed: list[KernelName] = []
    for kernel in KERNELS:
        writer = SummaryWriter(log_dir=log_root / kernel)
        ok, loss_value, taken = run_overfit(kernel, held_out, device, scaled, writer)
        print(
            f"overfit {kernel} done loss {loss_value:.4f} steps {taken} ok {ok}",
            flush=True,
        )
        writer.close()
        if ok:
            passed.append(kernel)
    if args.overfit:
        return
    rows: list[dict[str, str]] = []
    for kernel in KERNELS:
        if kernel not in passed:
            rows.append(
                {"kernel": kernel, **{str(length): "overfit_failed" for length in lengths}}
            )
            continue
        writer = SummaryWriter(log_dir=log_root / kernel)
        model = run_train(kernel, device, scaled, steps, batch_size, writer)
        row = {"kernel": kernel}
        for length in lengths:
            score = exact_match_length(model, length, device)
            row[str(length)] = f"{score:.4f}"
            writer.add_scalar("eval/exact_match", score, length)
            print(f"eval {kernel} length {length} exact {score:.4f}", flush=True)
        writer.close()
        rows.append(row)
        write_table(table_path, rows, lengths)


if __name__ == "__main__":
    main()
