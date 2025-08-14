import torch
from torch.utils.data import Dataset

"""
    Accepts sequences as (N, T, C) and converts to (N, C, T) for Conv1d.
    Targets are expected as integer-encoded labels.
"""

class SeqDataset(Dataset):
    def __init__(self, X_seq, y_int):
        self.X = torch.from_numpy(X_seq.transpose(0, 2, 1)).float()  # (N, C, T)
        self.y = torch.from_numpy(y_int).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]