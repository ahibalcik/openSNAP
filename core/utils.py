# core/utils.py
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sae import SparseAutoencoder

def Vecto2D(SAE_encoder, trained_model_path, trained_vectors):
    model = SAE_encoder
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    # Extract semantic vectors from input texts
    semantic_vectors = []
    for text in trained_vectors:
        semantic_vector = model.get_semantic_vector(text)
        semantic_vectors.append(semantic_vector)
    
    # Convert to tensor
    semantic_tensor = torch.stack(semantic_vectors)
    
    # Convert to numpy array
    semantic_array = semantic_tensor.numpy()

    # Project to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    data_2D = tsne.fit_transform(semantic_array)

    return data_2D

def get_dataframe(data_2D):
    data_list = []
    # 2D data to dataframe
    for i in range(len(data_2D)):
        data_dict = {
            'x': data_2D[i][0],
            'y': data_2D[i][1]
        }
        data_list.append(data_dict)
    
    # Create pandas dataframe
    df = pd.DataFrame(data_list)
    
    return df

def load_config(config_path: str):
    """
    Load configuration from a YAML file.
    Args:
        config_path: Path to YAML configuration file
    Returns:
        Dictionary containing configuration parameters
    """
    # Open and read YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Return loaded configuration
    return config

def save_csv(dataframe, path: str):
    """
    Save DataFrame as CSV file.
    Args:
        dataframe: pandas DataFrame to save
        path: save path (e.g. 'result/ko/data.csv')
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save DataFrame to CSV
    dataframe.to_csv(path, index=False, encoding='utf-8')