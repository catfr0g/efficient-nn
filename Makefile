.DEFAULT_GOAL := help

UV ?= uv
MEASUREMENTS ?= hw1/results/measurements.csv
KERNELS ?= hw1/results/kernels.csv
THETA ?= hw1/results/theta.json
FIGURES ?= hw1/results/figures
MEMORY_LIMIT_GIB ?= 4

.PHONY: help sync gpu-check measure calibrate hw1 lint \
	hw2-mqmtar hw2-mqmtar-scaled hw2-mqmtar-budget hw2-reverse hw2-sort hw2-regbench hw2-tb

help:
	@echo "sync gpu-check measure calibrate hw1 lint"
	@echo "hw2-mqmtar hw2-mqmtar-scaled hw2-mqmtar-budget hw2-reverse hw2-sort hw2-regbench hw2-tb"

sync:
	$(UV) sync

gpu-check:
	$(UV) run python hw1/gpu_check.py

measure:
	$(UV) run python -m hw1.measure --memory-limit-gib "$(MEMORY_LIMIT_GIB)" --measurements "$(MEASUREMENTS)" --kernels "$(KERNELS)"

calibrate:
	$(UV) run python -m hw1.calibrate --measurements "$(MEASUREMENTS)" --output "$(THETA)" --figures "$(FIGURES)"

hw1: measure calibrate

lint:
	$(UV) run ruff check hw1

hw2-mqmtar:
	$(UV) run python -m hw2.src.train

hw2-mqmtar-scaled:
	$(UV) run python -m hw2.src.train --scaled

hw2-mqmtar-budget:
	$(UV) run python -m hw2.src.train --budget

hw2-reverse:
	$(UV) run python -m hw2.src.train_reverse

hw2-sort:
	$(UV) run python -m hw2.src.train_reverse --task sort

hw2-regbench:
	$(UV) run python -m hw2.src.train_regbench

hw2-tb:
	$(UV) run tensorboard --logdir hw2/results/tb
