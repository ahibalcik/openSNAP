import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from core.models import MBartModelHandler
from collections import defaultdict
import os

class TextDataset(Dataset):
    def __init__(self, num_samples=10000):
        """
        Initialize a multilingual parallel text dataset
        Args:
            num_samples: Number of samples to sample
        """
        self.data = []
        
        # Load OPUS-100 dataset (English-centric multilingual parallel corpus)
        languages = ['ko', 'en', 'es', 'zh']
        data_pairs = {}
        
        # Load parallel data for each language pair
        print("Loading parallel data for each language pair...")
        for lang in languages:
            if lang != 'en':
                dataset = load_dataset('opus100', f'en-{lang}', split='train', streaming=True)
                data_pairs[f'en-{lang}'] = list(dataset.take(num_samples))

        # Create a dictionary with English sentences as keys
        print("Creating a dictionary...")
        en_sentences = defaultdict(dict)
        for pair in data_pairs:
            for example in data_pairs[pair]:
                en_text = example['translation']['en']
                other_lang = pair.split('-')[1]
                other_text = example['translation'][other_lang]
                en_sentences[en_text][other_lang] = other_text

        # Select sentences that have all four languages and store them as tuples
        print("Selecting sentences with all four languages...")
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
    def __init__(self, input_dim=1024, hidden_dim=128, sparsity_param=0.05, sparsity_weight=0.1, variance_threshold=0.95):
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
        """Forward pass through the autoencoder"""
        # Normalize input values
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, min=-1e6, max=1e6)
        
        encoded = self.encoder(x)
        encoded = torch.clamp(encoded, min=-1e6, max=1e6)
        
        reconstructed = self.decoder(encoded)
        reconstructed = torch.clamp(reconstructed, min=-1e6, max=1e6)
        
        return reconstructed, encoded
        
    def get_semantic_vector(self, x):
        """Extract the semantic vector of the input tensor"""
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded
    
    def loss_function(self, x, reconstructed, encoded):
        """Calculate the loss function"""
        mse_loss = nn.MSELoss()(reconstructed, x)
        
        # Add variance preservation constraint
        x_var = torch.var(x, dim=0)
        reconstructed_var = torch.var(reconstructed, dim=0)
        variance_ratio = torch.mean(reconstructed_var / x_var)
        variance_penalty = torch.abs(variance_ratio - self.variance_threshold)
        
        # Sparsity penalty
        avg_activation = torch.mean(encoded, dim=0)
        kl_div = self.sparsity_param * torch.log(self.sparsity_param / avg_activation) + \
                 (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / (1 - avg_activation))
        sparsity_penalty = torch.sum(kl_div)
        
        return mse_loss + self.sparsity_weight * sparsity_penalty + variance_penalty

def train_autoencoder(model_save_path='models/sae.pth', batch_size=32, epochs=10):
    """
    Autoencoder training function
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and data loader
    print("Initializing dataset...")
    dataset = TextDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize MBart model
    print("Initializing MBart model...")
    mbart_model = MBartModelHandler()
    
    # Initialize autoencoder model
    print("Initializing autoencoder model...")
    model = SparseAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    print("Starting training loop...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Extract texts for each language
            ko_texts = [item[0] for item in batch]  # Korean texts
            
            # Extract vectors using MBart
            vectors, _ = mbart_model.extract_activations(
                DataLoader([(text,) for text in ko_texts], batch_size=len(ko_texts)), 
                "ko_KR"
            )
            
            # Move tensors to device
            vectors = vectors.to(device)
            
            # Train autoencoder
            optimizer.zero_grad()
            reconstructed, encoded = model(vectors)
            loss = model.loss_function(vectors, reconstructed, encoded)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SparseAutoencoder().to(device)
    
    # Run model training
    train_autoencoder(model_save_path='models/sae.pth')