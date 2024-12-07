# core/utils.py
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sae import SparseAutoencoder

def Vecto2D(*args):
    """
    여러 언어의 128차원 단어 벡터 리스트들과 해당 텍스트들을 2차원으로 변환하는 함수
    
    Args:
        *args: (vectors, texts) 튜플들의 리스트
               vectors: 128차원 벡터들 (numpy array)
               texts: 원본 텍스트 리스트
        
    Returns:
        각 언어별 2차원으로 변환된 벡터와 텍스트 리스트들의 튜플
    """
    # 벡터와 텍스트 분리
    vectors_lists = [arg[0] for arg in args]
    texts_lists = [arg[1] for arg in args]
    
    # 모든 벡터들을 하나로 합침
    combined_vectors = np.concatenate(vectors_lists, axis=0)
    
    # 전체 벡터에 대해 t-SNE를 한번만 학습/적용
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_2d = tsne.fit_transform(combined_vectors)
    
    # 각 언어별 벡터 개수
    lengths = [len(vectors) for vectors in vectors_lists]
    
    # 변환된 2D 벡터들을 다시 언어별로 분리하고 텍스트와 함께 반환
    start_idx = 0
    result = []
    for length, texts in zip(lengths, texts_lists):
        end_idx = start_idx + length
        result.append((combined_2d[start_idx:end_idx], texts))
        start_idx = end_idx
        
    return tuple(result)

def get_dataframe(data_2D_with_text):
    """
    2D 벡터와 텍스트를 데이터프레임으로 변환
    
    Args:
        data_2D_with_text: (2D_vectors, texts) 튜플
    """
    vectors_2d, texts = data_2D_with_text
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