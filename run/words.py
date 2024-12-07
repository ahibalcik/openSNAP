import torch
from core.data_loader import get_dataloader
from core.models import MBartModelHandler
from core.sae import SparseAutoencoder
from core.utils import visualize_statistics, save_statistics_to_csv, Vecto2D, get_dataframe
import os

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

    try:
        # �과 저장 디렉토리 생성
        result_dir = "result"
        os.makedirs(result_dir, exist_ok=True)
        
        # 데이터 로드 및 전처리
        with torch.no_grad():
            vectors, texts = model.extract_activations(dataloader, "ko_KR")
        print("activations load")

        # GPU로 이동
        if torch.cuda.is_available():
            vectors = vectors.cuda()
            sae_model = sae_model.cuda()
        
        # SAE 모델을 통한 압축
        compressed_activations, _ = sae_model(vectors)
        
        # CPU로 이동 및 전처리
        compressed_activations = compressed_activations.cpu().detach()
        
        # 2D 벡터로 변환
        compressed_activations_2D = Vecto2D(
            (compressed_activations, texts)
        )
        
        # 데이터프레임으로 변환
        df = get_dataframe(compressed_activations_2D)
        
        # 통계 저장 (경로 수정)
        csv_path = os.path.join(result_dir, "statistics.csv")
        save_statistics_to_csv(df, csv_path)
        print(f"statistics saved to {csv_path}")
        
        # 통��� 그림 저장 (경로 수정)
        png_path = os.path.join(result_dir, "statistics.png")
        visualize_statistics(compressed_activations_2D, save=True, filename=png_path)
        print(f"statistics saved to {png_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

