"""
This module contains PyTorch model code for the HEP-CNN
RPV classifier.
"""

# Compatibility
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# System
import time

# Externals
import torch
import torch.nn as nn

# Locals
from base_trainer import BaseTrainer

class HEPCNNClassifier(nn.Module):
    """
    HEP-CNN RPV classifier model.
    """

    def __init__(self, input_shape, conv_sizes, dense_sizes, dropout):
        """HEP-CNN classifier constructor"""
        super(HEPCNNClassifier, self).__init__()

        # Add the convolutional layers
        conv_layers = []
        in_size = input_shape[0]
        for conv_size in conv_sizes:
            conv_layers.append(nn.Conv2d(in_size, conv_size,
                                         kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            in_size = conv_size
        self.conv_net = nn.Sequential(*conv_layers)

        # Add the dense layers
        dense_layers = []
        in_height = input_shape[1] // (2 ** len(conv_sizes))
        in_width = input_shape[2] // (2 ** len(conv_sizes))
        in_size = in_height * in_width * in_size
        for dense_size in dense_sizes:
            dense_layers.append(nn.Linear(in_size, dense_size))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_size = dense_size
        dense_layers.append(nn.Linear(in_size, 1))
        dense_layers.append(nn.Sigmoid())
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x):
        h = self.conv_net(x)
        h = h.view(h.size(0), -1)
        return self.dense_net(h).squeeze(-1)

class HEPCNNTrainer(BaseTrainer):
    """Trainer code for the HEP-CNN classifier."""

    def __init__(self, **kwargs):
        super(HEPCNNTrainer, self).__init__(**kwargs)

    def build_model(self, input_shape, conv_sizes, dense_sizes, dropout,
                    optimizer='Adam', learning_rate=0.001):
        """Instantiate our model"""
        self.model = HEPCNNClassifier(input_shape=input_shape,
                                      conv_sizes=conv_sizes,
                                      dense_sizes=dense_sizes,
                                      dropout=dropout).to(self.device)
        opt_type = dict(Adam=torch.optim.Adam)[optimizer]
        self.optimizer = opt_type(self.model.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.BCELoss()

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0
        start_time = time.time()
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.info('  Training loss: %.3f' % summary['train_loss'])
        return summary

    def evaluate(self, data_loader):
        """Evaluate the model"""
        with torch.no_grad():
            self.model.eval()
            summary = dict()
            sum_loss = 0
            sum_correct = 0
            start_time = time.time()
            # Loop over batches
            for i, (batch_input, batch_target) in enumerate(data_loader):
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                self.model.zero_grad()
                batch_output = self.model(batch_input)
                sum_loss += self.loss_func(batch_output, batch_target)
                # Count number of correct predictions
                preds, labels = batch_output > 0.5, batch_target > 0.5
                sum_correct += preds.eq(labels).sum().item()
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / len(data_loader.dataset)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary
