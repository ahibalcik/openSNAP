# stat/ko.py

import argparse
from core.models import MBartModelHandler
from core.sae import SparseAutoencoder
from core.data_loader import get_dataloader
from core.utils import load_config

def main():
    """
    Main function to perform statistical analysis on mBART-50 hidden representations.
    """

    ko_vectors = model.extract_activations(ko_dataloader, "ko_KR")
    en_vectors = model.extract_activations(en_dataloader, "en_XX")

    # 벡터와 원본 텍스트를 2D로 변환
    ko_2d, en_2d = Vecto2D(
        (ko_vectors, ko_texts),
        (en_vectors, en_texts)
    )

    # 데이터프레임 생성
    ko_df = get_dataframe(ko_2d)
    en_df = get_dataframe(en_2d)

    # 시각화
    plot_2d_vectors(
        dataframes=[ko_df, en_df],
        labels=['Korean', 'English'],
        colors=['#FF6B6B', '#4ECDC4'],
        figsize=(15, 10),
        save_path='result/vector_visualization.png'
    )

    # TODO: Parse arguments if needed   
    # TODO: Parse arguments if needed
    # TODO: Load configuration
    # TODO: Initialize model handler
    # TODO: Load dataset and create DataLoader
    # TODO: Initialize and train SAE
    # TODO: Perform statistical analysis on SAE features
    # TODO: Save or visualize results
    pass

if __name__ == "__main__":
    main()
