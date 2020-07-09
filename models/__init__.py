"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    elif name == 'hitgraphs':
        from .hitgraphs import get_datasets
        return get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'dummy':
        from .dummy import get_datasets
        train_dataset, valid_dataset = get_datasets(**data_args)
    elif name == 'hitgraphs':
        from .EC1 import hitgraphs
        train_dataset, valid_dataset = hitgraphs.get_datasets(**data_args)
        collate_fn = hitgraphs.collate_fn
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader

"""                                                                                
Python module for holding our PyTorch trainers.                                    
                                                                                   
Trainers here inherit from the BaseTrainer and implement the logic for             
constructing the model as well as training and evaluation.                         
"""

def get_trainer(name, **trainer_args):
    """                                                                            
    Factory function for retrieving a trainer.                                     
    """
    if name == 'hello':
        from .hello import HelloTrainer
        return HelloTrainer(**trainer_args)
    elif name == 'basic':
        from .basic import BasicTrainer
        return BasicTrainer(**trainer_args)
    elif name == 'gnn':
        from .trainers.gnn import GNNTrainer
        return GNNTrainer(**trainer_args)
    else:
        raise Exception('Trainer %s unknown' % name)


"""                                                                                
Python module for holding our PyTorch models.                                      
"""

def get_model(name, **model_args):
    """                                                                            
    Top-level factory function for getting your models.                            
    """
    if name == 'cnn_classifier':
        from .cnn_classifier import CNNClassifier
        return CNNClassifier(**model_args)
    elif name == 'gnn_segment_classifier':
        from models.EC1.ec1 import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    else:
        raise Exception('Model %s unknown' % name)
