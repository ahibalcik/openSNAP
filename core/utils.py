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

def Vecto2D(*args):
    """
    여러 언어의 터들을 2차원으로 변환하는 함수
    """
    # 벡터와 텍스트 분리
    vectors_lists = [arg[0] for arg in args]
    texts_lists = [arg[1] for arg in args]
    
    # numpy array로 변환 및 전처리
    vectors_arrays = []
    for vectors in vectors_lists:
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=1e6, neginf=-1e6)
        vectors_arrays.append(vectors)
    
    # 모든 벡터들을 하나로 합침
    combined_vectors = np.concatenate(vectors_arrays, axis=0)
    
    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_vectors)-1))
    combined_2d = tsne.fit_transform(combined_vectors)
    
    # 결과 분리
    start_idx = 0
    result = []
    for length, texts in zip([len(v) for v in vectors_arrays], texts_lists):
        end_idx = start_idx + length
        result.append((combined_2d[start_idx:end_idx], texts))
        start_idx = end_idx
        
    return tuple(result)

def get_dataframe(data_2D_with_text):
    """
    2D 벡터와 ���스트를 데이터프레임으로 변환
    
    Args:
        data_2D_with_text: (2D_vectors, texts) 튜플의 튜플
    """
    # 첫 번째 언어 쌍만 사용 (튜플의 첫 번째 요소)
    vectors_2d, texts = data_2D_with_text[0]
    data_list = []
    
    for i in range(len(vectors_2d)):
        data_dict = {
            'x': vectors_2d[i][0],
            'y': vectors_2d[i][1],
            'text': texts[i]
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

def visualize_statistics(data_2D_with_text, save=False, filename="statistics.png"):
    """
    Visualize 2D statistics and optionally save the figure.
    
    Args:
        data_2D_with_text: (2D_vectors, texts) 튜플의 튜플
        save: Save the figure if True
        filename: Filename to save the figure
    """
    plt.figure(figsize=(15, 10))
    
    # 첫 번째 언어 쌍만 시각화
    vectors_2d, texts = data_2D_with_text[0]
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    
    for i, text in enumerate(texts):
        plt.annotate(text, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=8)

    plt.title("2D Visualization of Compressed Activations")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Statistics visualized and saved to {filename}")
    
    plt.show()