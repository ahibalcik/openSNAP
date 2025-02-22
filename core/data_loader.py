# core/data_loader.py

import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")  # Dataset folder path constant

class TextDataset(Dataset):
    def __init__(self, file_name):
        self.data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                # Read tuple from line
                line = line.strip().strip('()')  # Remove parentheses
                translations = line.split(', ')  # Split by comma
                if len(translations) == 4:
                    self.data.append(tuple(translations))  # Store as tuple

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Return tuple (Korean, English, Spanish, Chinese)

def get_dataloader(file_name, tokenizer, batch_size=32, shuffle=True):
    dataset = TextDataset(file_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
