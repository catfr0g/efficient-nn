"""RegBench: accuracy последнего символа и TVD.

В статье ищут глубину и ширину и гоняют наборы от 1 до 40 тысяч примеров
до 200 эпох. Здесь один конфиг их маленькой Memory Mosaic: 2 слоя, ширина
128, 8 голов, 448 слотов, батч 32, lr 5e-4. 2000 обучающих автоматов,
256 тестовых, 4000 шагов. Строки короче, чем в бенчмарке, чтобы вход
умещался в 128 токенов.
"""

import csv
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import KernelName, MosaicConfig, StackAssoMem
from .regbench import PAD_TO, VOCAB_SIZE, Batch, collate, generate_example
from .train import KERNELS, _bind_train_step

TRAIN_EXAMPLES = 2_000
TEST_EXAMPLES = 256
STEPS = 4_000
BATCH_SIZE = 32
LOG_EVERY = 10
EVAL_EVERY = 1_000

PROTOCOL = (
    "RegBench vs paper Table 1: 2 layers, width 128, 8 heads, "
    "persistent slots 448, batch 32, lr 5e-4, dropout 0, one seed. "
    "2000 train automata and 256 test, 4000 steps, versus up to 40k "
    "examples and 200 epochs. Strings 8-12 of length 2-6 versus 10-20 "
    "of length 1-50. Alphabet 8, states 4-8."
)


def make_model(kernel: KernelName, device: torch.device) -> StackAssoMem:
    torch.manual_seed(0)
    config = MosaicConfig(
        vocab_size=VOCAB_SIZE,
        n_head=8,
        n_embd=128,
        n_layer=2,
        block_size=PAD_TO,
        pmem_size=448,
        ic_dropout=0.0,
        hd_dropout=0.0,
        kernel=kernel,
        kernel_topk=8,
    )
    model = StackAssoMem(config)
    model.to(device)
    return model


def optimizer(model: StackAssoMem, device: torch.device) -> torch.optim.Optimizer:
    return model.configure_optimizers(0.1, 5e-4, (0.9, 0.95), device.type)


def build_split(seed: int, count: int) -> list[tuple[list[int], list[int], list[list[int]]]]:
    rng = random.Random(seed)
    return [generate_example(rng) for _ in range(count)]


def sample_batch(
    examples: list[tuple[list[int], list[int], list[list[int]]]],
    step: int,
    device: torch.device,
) -> Batch:
    rng = random.Random(step)
    chosen = [examples[rng.randrange(len(examples))] for _ in range(BATCH_SIZE)]
    return collate(chosen, device)


@torch.no_grad()
def evaluate(model: StackAssoMem, batches: list[Batch]) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    correct = 0
    rows = 0
    tvd_sum = 0.0
    tvd_count = 0
    for batch in batches:
        logits = model(batch.idx, logits_only=True)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits_only did not return logits")
        supervised = batch.targets >= 0
        positions = torch.arange(batch.targets.size(1), device=batch.targets.device)
        last = torch.where(supervised, positions.view(1, -1), -1).amax(dim=-1)
        predicted = logits.argmax(dim=-1)
        gold = batch.targets.gather(1, last[:, None]).squeeze(1)
        chosen = predicted.gather(1, last[:, None]).squeeze(1)
        correct += int((chosen == gold).sum())
        rows += int(gold.numel())
        probs = F.softmax(logits, dim=-1)
        distance = 0.5 * (probs - batch.legal).abs().sum(dim=-1)
        tvd_sum += float((distance * supervised).sum())
        tvd_count += int(supervised.sum())
    if was_training:
        model.train()
    return correct / rows, tvd_sum / tvd_count


def test_batches(examples: list[tuple[list[int], list[int], list[list[int]]]], device: torch.device) -> list[Batch]:
    batches: list[Batch] = []
    for start in range(0, len(examples), BATCH_SIZE):
        batches.append(collate(examples[start : start + BATCH_SIZE], device))
    return batches


def run_train(
    kernel: KernelName,
    train_set: list[tuple[list[int], list[int], list[list[int]]]],
    held_out: list[Batch],
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[float, float]:
    model = make_model(kernel, device)
    opt = optimizer(model, device)
    model.train()
    train_step = _bind_train_step(model, opt)
    if next(model.parameters()).is_cuda:
        print(f"compiling {kernel}", flush=True)
    progress = tqdm(range(1, STEPS + 1), desc=f"train {kernel}")
    accuracy = 0.0
    tvd = 0.0
    for step in progress:
        batch = sample_batch(train_set, step, device)
        loss = train_step(batch.idx, batch.targets)
        loss_value = float(loss)
        progress.set_postfix(loss=f"{loss_value:.3f}")
        if step % LOG_EVERY == 0:
            writer.add_scalar("train/loss", loss_value, step)
        if step % EVAL_EVERY == 0 or step == STEPS:
            accuracy, tvd = evaluate(model, held_out)
            writer.add_scalar("eval/accuracy", accuracy, step)
            writer.add_scalar("eval/tvd", tvd, step)
            model.train()
            print(f"eval {kernel} step {step} accuracy {accuracy:.4f} tvd {tvd:.4f}", flush=True)
    return accuracy, tvd


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kernel", "accuracy", "tvd"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(PROTOCOL, flush=True)
    print(f"device {device}", flush=True)
    train_set = build_split(0, TRAIN_EXAMPLES)
    held_out = test_batches(build_split(1, TEST_EXAMPLES), device)
    log_root = Path("hw2/results/tb/regbench")
    table_path = Path("hw2/results/regbench_table.csv")
    rows: list[dict[str, str]] = []
    for kernel in KERNELS:
        writer = SummaryWriter(log_dir=log_root / kernel)
        accuracy, tvd = run_train(kernel, train_set, held_out, device, writer)
        writer.close()
        rows.append({"kernel": kernel, "accuracy": f"{accuracy:.4f}", "tvd": f"{tvd:.4f}"})
        write_table(table_path, rows)


if __name__ == "__main__":
    main()
