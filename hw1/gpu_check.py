# %%
"""Check the Python, CUDA, and GPU environment for Homework 1."""

# %%
import sys

import matplotlib
import numpy as np
import pandas as pd
import pynvml
import scipy
import torch

print(f"Python      {sys.version.split()[0]}")
print(f"numpy       {np.__version__}")
print(f"pandas      {pd.__version__}")
print(f"matplotlib  {matplotlib.__version__}")
print(f"scipy       {scipy.__version__}")
print(f"torch       {torch.__version__}")
print(f"CUDA build  {torch.version.cuda}")
print(f"CUDA ready  {torch.cuda.is_available()}")

# %%
assert torch.cuda.is_available(), "CUDA is not available — check the GPU driver and the PyTorch wheel"

device = torch.device("cuda")
print(torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device=device)
y = x @ x
torch.cuda.synchronize()
print("gpu matmul ok", float(y[0, 0]))
print(f"allocated {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")

# %%
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
print("driver", pynvml.nvmlSystemGetDriverVersion())
print(f"power  {pynvml.nvmlDeviceGetPowerUsage(handle) / 1000:.1f} W")
