"""
Hello world PyTorch trainer.
"""

# System
import time

# Locals
from .base_trainer import BaseTrainer

class HelloTrainer(BaseTrainer):
    """Hello world trainer object"""

    def __init__(self, **kwargs):
        super(HelloTrainer, self).__init__(**kwargs)

    def build_model(self):
        pass
    def write_checkpoint(self, checkpoint_id):
        pass
    def write_summaries(self):
        pass

    def print_model_summary(self):
        self.logger.info('Hello world')

    def train_epoch(self, data_loader):
        summary = dict()
        start_time = time.time()
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.logger.info('  Train batch input %s, target %s',
                             batch_input.size(), batch_target.size())
        self.logger.info('  Processed %i training batches' % (i + 1))
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = 0
        return summary

    def evaluate(self, data_loader):
        """"Evaluate the model"""
        summary = dict()
        start_time = time.time()
        # Loop over validation batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.logger.info('  Valid batch input %s, target %s',
                             batch_input.size(), batch_target.size())
        self.logger.info('  Processed %i validation batches' % (i + 1))
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = 0
        summary['valid_acc'] = 1
        return summary
