"""
Dataset handling code for the HEP-CNN classifier
"""

# Externals
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

def load_file(filename, n_samples):
    """Load one file from the dataset"""
    with h5py.File(filename, 'r') as f:
        data_group = f['all_events']
        data = data_group['hist'][:n_samples][:,None,:,:].astype(np.float32)
        labels = data_group['y'][:n_samples].astype(np.float32)
        weights = data_group['weight'][:n_samples].astype(np.float32)
    return data, labels, weights

class HEPDataset(Dataset):
    """PyTorch Dataset for the HEP-CNN images"""
    
    def __init__(self, input_file, n_samples):
        x, y, w = load_file(input_file, n_samples)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.x.size(0)
