# core/utils.py
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from core.sae import SparseAutoencoder

def Vecto2D(data):
    """
    Convert vectors to 2D (e.g., apply TSNE)
    
    Args:
        data: (vectors, texts) tuple
    Returns:
        compressed_2d: 2D converted vectors
        texts: input texts
    """
    vectors = data
    # Apply TSNE
    tsne = TSNE(n_components=2)
    compressed_2d = tsne.fit_transform(vectors.numpy())  # Convert vectors to numpy array and apply TSNE
    return compressed_2d

def get_dataframe(data_2D_with_text, lang_code='en'):
    """
    Convert 2D vectors and texts to a dataframe
    
    Args:
        data_2D_with_text: ((2D_vectors, texts), ...) tuple of tuples
        lang_code: Language code (e.g., 'en', 'ko', 'zh', 'es')
    """
    vectors_2d, texts = data_2D_with_text  # unpacking modified
    data_list = []
    
    # Check length
    if len(vectors_2d) != len(texts):
        print(f"Warning: Length mismatch between vectors and texts for language {lang_code}.")
        return pd.DataFrame()  # Return empty dataframe
    
    for i in range(len(vectors_2d)):
        data_dict = {
            'x': vectors_2d[i][0],
            'y': vectors_2d[i][1],
            'text': texts[i],  # Language code removed
            'lang': lang_code  # Language distinction column added
        }
        data_list.append(data_dict)
    
    return pd.DataFrame(data_list)

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

def save_statistics_to_csv(dataframe, path: str):
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
    print(f"Statistics saved to {path}")

def visualize_statistics(data_2D_with_texts, save=False, filename="statistics.png"):
    """
    Visualize 2D statistics with different colors for each language
    
    Args:
        data_2D_with_texts: [(2D_vectors, texts, lang_code), ...] list
        save: Save the figure if True
        filename: Filename to save the figure
    """
    plt.figure(figsize=(15, 10))
    
    # Define colors for each language
    colors = {
        'en': '#1f77b4',  # Blue
        'ko': '#ff7f0e',  # Orange
        'zh': '#2ca02c',  # Green
        'es': '#d62728'   # Red
    }
    
    # Visualize each language's data
    for compressed_2d, texts, lang_code in data_2D_with_texts:
        
        # Convert compressed_2d to numpy array if it's a list
        compressed_2d = np.array(compressed_2d)  # Convert to numpy array
        
        plt.scatter(
            compressed_2d[:, 0], 
            compressed_2d[:, 1], 
            c=colors[lang_code],
            label=f'Language: {lang_code}',
            alpha=0.6
        )
        
        # Add text labels
        for i, text in enumerate(texts):
            plt.annotate(
                text,
                (compressed_2d[i, 0], compressed_2d[i, 1]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )

    plt.title("2D Visualization of Compressed Activations by Language")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Statistics visualized and saved to {filename}")
    
    plt.show()