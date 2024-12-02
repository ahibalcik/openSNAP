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
    model.load_state_dict(torch.load(trained_model_path)) # model path
    model.eval()

    # Change data to tensor
    data_tensor = torch.FloatTensor(trained_vectors)

    # Take incoded data
    with torch.no_grad():
        encoded_data = model.encode(data_tensor)

    # Change to numpy array
    encoded_array = encoded_data.numpy()

    # t-SNE to project in 2D
    tsne = TSNE(n_components=2, random_state=0)
    data_2D = tsne.fit_transform(encoded_array)

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
    """
    # TODO: Implement configuration loading
    pass

def save_model(model, path: str):
    """
    Save the trained model to a specified path.
    """
    # TODO: Implement model saving
    pass

def load_model(path: str):
    """
    Load a trained model from a specified path.
    """
    # TODO: Implement model loading
    pass
