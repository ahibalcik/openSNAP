import torch
from core.data_loader import get_dataloader
from core.models import MBartModelHandler
from core.sae import SparseAutoencoder
from utils import visualize_statistics, save_statistics_to_csv

def main():
    # 데이터셋 로드
    dataloader = get_dataloader("words_datasets.txt", MBartModelHandler().tokenizer, batch_size=32)

    # mBART 모델 초기화
    model = MBartModelHandler()

    # SAE 모델 초기화
    sae_model = SparseAutoencoder()

    # 데이터 로드 및 전처리
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            # mBART 모델 forward pass
            _, texts = model.extract_activations(dataloader, "ko_KR")
            activations.extend(texts)

    # SAE 모델을 통한 스파스 압축
    compressed_activations = []
    for activation in activations:
        compressed_activation = sae_model(activation)
        compressed_activations.append(compressed_activation)

    # 통계 계산 및 시각화
    statistics = visualize_statistics(compressed_activations)
    save_statistics_to_csv(statistics, "statistics.csv")

    # 통계 그림 저장
    visualize_statistics(compressed_activations, save=True, filename="statistics.png")

if __name__ == "__main__":
    main()

