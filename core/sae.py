# core/sae.py

import torch
import torch.nn as nn
from models import MBartModelHandler

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, lambda_reg: float):
        """
        Initialize the Sparse Autoencoder with encoder and decoder layers.
        """
        super(SparseAutoencoder, self).__init__()
        self.model = MBartModelHandler()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Linear(768, feature_dim)
        )
        
        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.ReLU(), 
            nn.Linear(768, input_dim)
        )

    def encode(self, x):
        """
        Encode input tensor into semantic space vector
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode encoded vector back to original dimension
        """
        return self.decoder(z)

    def get_semantic_vector(self, input_text: str):
        """
        Return semantic vector for input text
        Args:
            input_text: Input text to encode
        Returns:
            Semantic vector representation of input text
        """
        # Extract activation from mBART model
        activation = self.model.extract_activations(input_text, "ko_KR")
        
        # Encode extracted activation to generate semantic vector
        with torch.no_grad():
            semantic_vector = self.encode(activation)
            
        return semantic_vector