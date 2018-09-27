"""
Using this to test the data DistributedSampler of PyTorch.
"""

import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler 

# Setup logging
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# Initialize MPI
dist.init_process_group(backend='mpi')
logging.info('MPI rank %i' % dist.get_rank())

# Define our "dataset"
x = torch.Tensor(np.arange(100))

# Distributed sampler and data loader
sampler = DistributedSampler(x)
loader = DataLoader(x, batch_size=10, sampler=sampler)

# Loop over batches
for bx in loader:
    print(bx)
