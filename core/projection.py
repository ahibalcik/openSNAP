import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import SAE_encoder

model = #SAE_encoder()
model.load_state_dict(torch.load('path_to_trained_model.pth')) # model path

model.eval()

# 
trained_vectors = np.random.rand(100, 50)  # trained vector

# Change data to tensor
data_tensor = torch.FloatTensor(trained_vectors)

# Take incoded data
with torch.no_grad():
    encoded_data = model.encode(data_tensor)

# Change to numpy array
encoded_array = encoded_data.numpy()

# t-SNE to project in 2D
tsne = TSNE(n_components=2, random_state=0)
low_dimensional_data = tsne.fit_transform(encoded_array)

# visualize
plt.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1])
plt.title('t-SNE Projection of Encoded Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()