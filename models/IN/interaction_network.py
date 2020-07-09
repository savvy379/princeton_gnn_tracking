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
    def __init__(self, object_dim, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()
        
        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, 150)
        self.object_model = ObjectModel(object_dim + effect_dim, 100)
        
    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        N = len(objects)
    
        senders   = [torch.matmul(objects[i].t(), sender_relations[i]) 
                     for i in range(N)]
        receivers = [torch.matmul(objects[i].t(), receiver_relations[i]) 
                     for i in range(N)]
        
        interaction_terms = [torch.cat([senders[i], receivers[i], relation_info[i]]) 
                             for i in range(N)]
        
        effects = self.relational_model(interaction_terms)
        
        effects_receivers = [torch.matmul(receiver_relations[i], effects[i]).t() 
                             for i in range(N)]
        aggregated = [torch.cat([objects[i].t(), effects_receivers[i]]) 
                      for i in range(N)]
        
        predicted = self.object_model(aggregated)
        predicted = [predicted[i].t() for i in range(N)]
        
        senders = [torch.matmul(predicted[i], sender_relations[i]) 
                   for i in range(N)]
        receivers = [torch.matmul(predicted[i], receiver_relations[i]) 
                     for i in range(N)]
        
        interaction_terms = [torch.cat([senders[i], receivers[i], relation_info[i]])
                             for i in range(N)]
        
        predicted = self.relational_model(interaction_terms)
        return predicted
