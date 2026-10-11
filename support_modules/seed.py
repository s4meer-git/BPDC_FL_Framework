import random
import numpy as np
import torch


# Receives the experiment seed from main.py and applies it to Python, NumPy and PyTorch random generators.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
