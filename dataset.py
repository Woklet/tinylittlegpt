import numpy as np
import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, data_path, block_size):
        data = np.fromfile(data_path, dtype=np.uint16)
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y
    @property
    def total_tokens(self):
        return self.data.numel()
