import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from models import MBartModelHandler
from collections import defaultdict
import os

class TextDataset(Dataset):
    def __init__(self, num_samples=10000 * 100):
        """
        다국어 병렬 텍스트 데이터셋 초기화
        Args:
            num_samples: 샘플링할 데이터 수
        """
        self.data = []
        
        # OPUS-100 데이터셋 로드 (영어 중심 다국어 병렬 코퍼스)
        languages = ['ko', 'en', 'es', 'zh']
        data_pairs = {}
        
        # 각 언어쌍의 병렬 데이터 로드
        for lang in languages:
            if lang != 'en':
                dataset = load_dataset('opus100', f'en-{lang}', split='train', streaming=True)
                data_pairs[f'en-{lang}'] = list(dataset.take(num_samples))

        # 영어 문장을 키로 하는 딕셔너리 생성
        en_sentences = defaultdict(dict)
        for pair in data_pairs:
            for example in data_pairs[pair]:
                en_text = example['translation']['en']
                other_lang = pair.split('-')[1]
                other_text = example['translation'][other_lang]
                en_sentences[en_text][other_lang] = other_text

        # 4개 언어가 모두 있는 문장만 선택하여 튜플로 저장
        self.data = []
        for en_text, translations in en_sentences.items():
            if all(lang in translations for lang in ['ko', 'es', 'zh']):
                self.data.append((
                    translations['ko'],  # Korean first for MBartModelHandler
                    en_text,            # English second
                    translations['es'],  # Spanish third 
                    translations['zh']   # Chinese fourth
                ))
                if len(self.data) >= num_samples:
                    break

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SparseAutoencoder(nn.Module):
    def __init__(self, model_save_path='models/sae.pth', input_dim=1024, hidden_dim=128, sparsity_param=0.05, sparsity_weight=0.1, variance_threshold=0.95):
        """
        다국어 희소 오토인코더 초기화
        Args:
            input_dim: 입력 차원 (mBART hidden state 크기)
            hidden_dim: 인코더의 은닉층 차원
            sparsity_param: 희소성 파라미터 (ρ)
            sparsity_weight: 희소성 패널티 가중치 (β)
            variance_threshold: 분산 보존 임계값 (95%)
        """
        super(SparseAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        
        self.sparsity_param = sparsity_param
        self.sparsity_weight = sparsity_weight
        self.variance_threshold = variance_threshold

        # 저장된 모델이 있으면 불러오기
        if os.path.exists(model_save_path):
            self.load_state_dict(torch.load(model_save_path))
            print(f"저장된 모델을 {model_save_path}에서 불러왔습니다.")
        else:
            print("저장된 모델이 없어 새로 학습을 시작합니다.")
            # 학습을 여기서 시작하지 않고, main 블록에서 처리

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded
        
    def get_semantic_vector(self, x):
        """입력 텐서의 의미 벡터 추출"""
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded
    
    def loss_function(self, x, reconstructed, encoded):
        """손실 함수 계산"""
        mse_loss = nn.MSELoss()(reconstructed, x)
        
        # 분산 보존 제약 추가
        x_var = torch.var(x, dim=0)
        reconstructed_var = torch.var(reconstructed, dim=0)
        variance_ratio = torch.mean(reconstructed_var / x_var)
        variance_penalty = torch.abs(variance_ratio - self.variance_threshold)
        
        # 희소성 패널티
        avg_activation = torch.mean(encoded, dim=0)
        kl_div = self.sparsity_param * torch.log(self.sparsity_param / avg_activation) + \
                 (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / (1 - avg_activation))
        sparsity_penalty = torch.sum(kl_div)
        
        return mse_loss + self.sparsity_weight * sparsity_penalty + variance_penalty

def train_autoencoder(model_save_path='models/sae.pth', batch_size=32, epochs=10):
    """
    오토인코더 학습 함수
    Args:
        model_save_path: 모델 저장 경로
        batch_size: 배치 크기
        epochs: 학습 에포크 수
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋과 데이터로더 초기화
    dataset = TextDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    model = SparseAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 학습 루프
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            reconstructed, encoded = model(batch)
            loss = model.loss_function(batch, reconstructed, encoded)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    # 모델 저장
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 초기화
    model = SparseAutoencoder().to(device)
    
    # 모델 학습 실행
    train_autoencoder(model_save_path='models/sae.pth')