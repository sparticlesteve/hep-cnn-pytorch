"""
Common PyTorch trainer code.
"""

# System
import os
import logging

# Externals
import numpy as np
import torch
import torch.distributed as dist

class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, device='cpu'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.output_dir = os.path.expandvars(output_dir)
        self.summaries = {}

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel()
             for p in self.model.parameters()))
        )

    def save_summary(self, summaries):
        """Save summary information"""
        for (key, val) in summaries.items():
            summary_vals = self.summaries.get(key, [])
            self.summaries[key] = summary_vals + [val]

    def write_summaries(self):
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir, 'summaries.npz')
        self.logger.info('Saving summaries to %s' % summary_file)
        np.savez(summary_file, **self.summaries)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(dict(model=self.model.state_dict()),
                   os.path.join(checkpoint_dir, checkpoint_file))

    def sync_params(self):
        """Broadcast model params from rank 0 to all ranks"""
        for param in self.model.parameters():
            dist.broadcast(param.data, 0)

    def sync_grads(self):
        """Aggregate gradients from all ranks"""
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data)
            param.grad.data /= size

    def build_model(self):
        """Virtual method to construct the model"""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Loop over epochs
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)
            summary = dict(epoch=i)
            # Train on this epoch
            summary.update(self.train_epoch(train_data_loader))
            # Evaluate on this epoch
            if valid_data_loader is not None:
                summary.update(self.evaluate(valid_data_loader))
            # Save summary, checkpoint
            self.save_summary(summary)
            self.write_checkpoint(checkpoint_id=i)

        return self.summaries
