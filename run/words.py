import torch
from core.data_loader import get_dataloader
from core.models import MBartModelHandler
from core.sae import SparseAutoencoder
from core.utils import visualize_statistics, save_statistics_to_csv, Vecto2D, get_dataframe

def main():
    # 데이터셋 로드
    dataloader = get_dataloader("words_datasets.txt", MBartModelHandler().tokenizer, batch_size=32)
    print("data loader loaded")

    # mBART 모델 초기화
    model = MBartModelHandler()
    print("mBART model loaded")

    # SAE 모델 초기화
    sae_model = SparseAutoencoder()
    print("sae_model loaded")

    # 데이터 로드 및 전처리
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            # mBART 모델 forward pass
            _, texts = model.extract_activations(dataloader, "ko_KR")
            activations.extend(texts)
    print("activations load")

    # SAE 모델을 통한 스파스 압축
    compressed_activations = []
    for activation in activations:
        compressed_activation = sae_model(activation)
        compressed_activations.append(compressed_activation)
    print("sae forward passed")

    # 2D 벡터로 변환
    compressed_activations_2D = Vecto2D(*[(compressed_activation, text) for compressed_activation, text in zip(compressed_activations, texts)])
    
    # 데이���프레임으로 변환
    df = get_dataframe(compressed_activations_2D)
    
    # 통계 저장
    save_statistics_to_csv(df, "statistics.csv")
    print("statistics saved to csv")
    
    # 통계 그림 저장
    visualize_statistics(compressed_activations_2D, save=True, filename="statistics.png")
    print("statistics saved to visual")

if __name__ == "__main__":
    main()

