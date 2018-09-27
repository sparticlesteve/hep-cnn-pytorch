"""
Distributed data-parallel training of the HEP-CNN-lite RPV Classifier
implemented in PyTorch.
"""

# Compatibility
from __future__ import division

# System
import argparse
import logging

# Externals
import numpy as np
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd

# Locals
from data import HEPDataset
from hepcnn import HEPCNNTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='config.yaml')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Initialize horovod and MPI
    hvd.init()
    logging.info('MPI rank %i' % hvd.rank())

    # Load configuration file
    with open(args.config) as f:
        config = yaml.load(f)
    if hvd.rank() == 0:
        logging.info('Configuration: %s' % config)

    # Load the data
    data_config = config['data_config']
    train_config = config['training_config']
    batch_size = train_config.pop('batch_size')
    train_dataset = HEPDataset(data_config['train_file'], n_samples=data_config['n_train'])
    valid_dataset = HEPDataset(data_config['valid_file'], n_samples=data_config['n_valid'])
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=hvd.size(),
                                       rank=hvd.rank())
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   sampler=train_sampler)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
    logging.info('Loaded data with shape: %s' % str(train_dataset.x.size()))

    # Instantiate the trainer
    model_config = config['model_config']
    learning_rate = model_config.pop('learning_rate') * hvd.size()
    input_shape = train_dataset.x.size()[1:]
    trainer = HEPCNNTrainer(output_dir=config['output_dir'], distributed=True)
    trainer.build_model(input_shape=input_shape, learning_rate=learning_rate,
                        **model_config)
    if hvd.rank() == 0:
        trainer.print_model_summary()

    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **train_config)
    if hvd.rank() == 0:
        trainer.write_summaries()

    # Print some conclusions
    n_train_samples = len(train_data_loader.sampler)
    n_valid_samples = len(valid_data_loader.sampler)
    logging.info('Finished training')
    train_time = np.mean(summary['train_time'])
    logging.info('Train samples %g time %gs rate %g samples/s' % (
        n_train_samples, train_time, n_train_samples / train_time))
    valid_time = np.mean(summary['valid_time'])
    logging.info('Valid rate: %g samples/s' % (n_valid_samples / valid_time))

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
