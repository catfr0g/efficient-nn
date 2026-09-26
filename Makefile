.DEFAULT_GOAL := help

UV ?= uv
MEASUREMENTS ?= hw1/results/measurements.csv
KERNELS ?= hw1/results/kernels.csv
THETA ?= hw1/results/theta.json
FIGURES ?= hw1/results/figures
MEMORY_LIMIT_GIB ?= 4

.PHONY: help sync gpu-check measure calibrate hw1 lint

help:
	@echo "sync gpu-check measure calibrate hw1 lint"

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
