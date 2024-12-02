# core/utils.py
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sae import SparseAutoencoder

def Vecto2D(SAE_encoder, trained_model_path, trained_vectors)
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

    # Check the shape of projected data
    print(data_2D.shape)
    # visualize
    plt.scatter(data_2D[:, 0], data_2D[:, 1])
    plt.title('t-SNE Projection of Encoded Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    return data_2D

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
