import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from models.IN.relational_model import RelationalModel
from models.IN.object_model import ObjectModel

class InteractionNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim, effect_dim, n_recur=1):
        super(InteractionNetwork, self).__init__()
        
        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, 150)
        self.object_model = ObjectModel(object_dim + effect_dim, 100)
        self.n_recur = n_recur

    def forward(self, objects, sender_relations, receiver_relations, relation_info):

        batch_size = len(objects)
        reembeded_objects = objects

        for i in range(self.n_recur):
            # marshalling step: build interaction terms
            senders   = [torch.matmul(reembeded_objects[i].t(), sender_relations[i]) 
                         for i in range(batch_size)]
            receivers = [torch.matmul(reembeded_objects[i].t(), receiver_relations[i]) 
                         for i in range(batch_size)]
            interaction_terms = [torch.cat([senders[i], receivers[i], relation_info[i]]) 
                                 for i in range(batch_size)]
        
            # relational model: determine effects
            effects = self.relational_model(interaction_terms)
        
            # aggregation step: aggregate effects
            effects_receivers = [torch.matmul(receiver_relations[i], effects[i]).t() 
                                 for i in range(batch_size)]
            aggregated = [torch.cat([reembeded_objects[i].t(), effects_receivers[i]]) 
                          for i in range(batch_size)]
        
            # object model: re-embed hit features
            reembeded_objects = self.object_model(aggregated)

        # re-marshalling step: build new interaction terms
        senders = [torch.matmul(reembeded_objects[i].t(), sender_relations[i]) 
                   for i in range(batch_size)]
        receivers = [torch.matmul(reembeded_objects[i].t(), receiver_relations[i]) 
                     for i in range(batch_size)]
        interaction_terms = [torch.cat([senders[i], receivers[i], relation_info[i]])
                             for i in range(batch_size)]
        edge_weights = self.relational_model(interaction_terms)

        return edge_weights
