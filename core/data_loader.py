# core/data_loader.py

import os
import ast
from torch.utils.data import Dataset, DataLoader

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")  # Dataset 폴더 경로 상수

class TextDataset(Dataset):
    def __init__(self, file_name: str, tokenizer):
        """
        Initialize text dataset.
        
        Args:
            file_name: Dataset filename (e.g. "words_datasets.txt")
            tokenizer: Tokenizer object
        """
        self.file_path = os.path.join(DATASET_DIR, file_name)
        self.tokenizer = tokenizer
        self.data = self._load_data()
        
    def _load_data(self):
        """Read dataset file and convert to list of tuples"""
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Convert string to tuple
                tuple_data = ast.literal_eval(line.strip())
                data.append(tuple_data)
        return data

    def __len__(self):
        """Return size of dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Return data sample for given index"""
        return self.data[idx]

def get_dataloader(file_name: str, tokenizer, batch_size: int, shuffle: bool = True):
    """
    Create data loader.
    
    Args:
        file_name: Dataset filename
        tokenizer: Tokenizer object
        batch_size: Batch size
        shuffle: Whether to shuffle data
    """
    dataset = TextDataset(file_name, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
