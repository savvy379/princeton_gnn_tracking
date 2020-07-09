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

class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, aggregated):
        N = len(aggregated)
        aggregated = [aggregated[i].t() for i in range(N)]
        influence = [self.layers(aggregated[i]) for i in range(N)]
        return influence

