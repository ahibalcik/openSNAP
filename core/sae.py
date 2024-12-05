import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from models import MBartModelHandler
import os

class TextDataset(Dataset):
    def __init__(self, num_samples=10000):
        """
        다국어 텍스트 데이터셋 초기화
        Args:
            num_samples: 샘플링할 데이터 수
        """
        # OSCAR 데이터셋에서 한국어, 영어, 스페인어, 중국어 샘플 로드
        languages = ['ko', 'en', 'es', 'zh']
        self.data = []
        
        for lang in languages:
            dataset = load_dataset('oscar', f'unshuffled_deduplicated_{lang}', streaming=True)
            samples = list(dataset['train'].take(num_samples // len(languages)))
            texts = [sample['text'] for sample in samples]
            # 각 언어별 텍스트를 튜플로 만들어 저장
            for text in texts:
                self.data.append((text, "", "", ""))  # ko, en, es, zh 순서로 저장
                
        # MBartModelHandler 초기화
        self.model_handler = MBartModelHandler()
        self.dataloader = DataLoader(self, batch_size=32, shuffle=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, sparsity_param=0.05, sparsity_weight=0.1, variance_threshold=0.95):
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
    # 모델 학습 실행
    train_autoencoder()


