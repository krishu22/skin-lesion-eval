import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.seed import set_seed
import numpy as np
import torch

set_seed(42)
print(np.random.rand(3))
print(torch.rand(3))

set_seed(42)
print(np.random.rand(3))   # should print the exact same 3 numbers as above
print(torch.rand(3))       # should print the exact same 3 numbers as above