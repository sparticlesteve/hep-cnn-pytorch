"""
Using this to test Horovod in PyTorch.
"""

import logging
import horovod.torch as hvd

# Setup logging
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# Initialize horovod/MPI
hvd.init()
logging.info('Horovod rank %i size %i' % (hvd.rank(), hvd.size()))
