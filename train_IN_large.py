import os
import random
import time
import argparse

import yaml
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

from models.IN.interaction_network import InteractionNetwork
from models.IN.graph import Graph, save_graphs, load_graph

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_IN.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/train_IN.yaml')
    return parser.parse_args()

def get_graphs(d):
    """ get all graphs from a directory
          return [("event<id1>", graph1, size), ...]
    """
    files = os.listdir(d)
    return [d+f for f in files]

def get_inputs(graphs):
    size = len(graphs)
    O = [Variable(torch.FloatTensor(graphs[i].X))
         for i in range(size)]
    Rs = [Variable(torch.FloatTensor(graphs[i].Ro))
          for i in range(size)]
    Rr = [Variable(torch.FloatTensor(graphs[i].Ri))
          for i in range(size)]
    #Rr = [torch.add(Variable(torch.FloatTensor(graphs[i].Ri)), Rs[i])
    #      for i in range(size)]
    #Rs = [Rr[i] for i in range(size)]
    Ra = [Variable(torch.FloatTensor(graphs[i].a)).unsqueeze(0)
          for i in range(size)]
    y  = [Variable(torch.FloatTensor(graphs[i].y)).unsqueeze(0).t()
          for i in range(size)]
    return O, Rs, Rr, Ra, y

def plotLosses(test_losses, train_losses, name):
    plt.plot(test_losses,  color='mediumslateblue', label="Testing",
             marker='h', lw=0, ms=1.4)
    plt.plot(train_losses, color='mediumseagreen',  label="Training",
             marker='h', lw=0, ms=1.4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(name, dpi=1200)
    plt.clf()

# grab config parameters
args = parse_args()
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)

graph_indir  = config['graph_indir']
model_outdir = config['model_outdir']
plot_outdir  = config['plot_outdir']
verbose      = config['verbose']
prep, pt_cut = config['prep'], config['pt_cut']
n_epoch      = config['n_epoch'] 
batch_size   = config['batch_size']
n_recur      = config['n_recur']
save_every   = config['save_every_n_epoch']
save_last    = config['save_last_n_epoch']
phi_reflect  = config['phi_reflect']
tag          = config['tag']

# job name, ex. "LP_0p5_1200"
job_name = "{0}_{1}_{2}_{3}".format(prep, pt_cut, n_epoch, tag)

if (verbose):
    print('\nBeginning', job_name)
    print(' ... writing models to', model_outdir)
    print(' ... writing plots to', plot_outdir)

# pull 1000 graphs from the train_1 sample
graph_dir = "{0}/IN_{1}_{2}/".format(graph_indir, prep, pt_cut)
graphs = get_graphs(graph_dir)
if (verbose): print(" ... n_graphs={0}".format(len(graphs)))

# objects: (r, phi, z); relations: (0); effects: (weight) 
object_dim, relation_dim, effect_dim = 3, 1, 1
interaction_network = InteractionNetwork(object_dim, relation_dim, effect_dim, n_recur)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(interaction_network.parameters(), lr=0.001)

# config mini-batch
train_size, test_size = 800, 200

n_batch = int(float(train_size)/batch_size)
if (verbose): print(" ... batch_size={0}".format(batch_size))

# prepare test graphs
#test_O, test_Rs, test_Rr, test_Ra, test_y = get_inputs(graphs[train_size:])
#if (verbose): print(" ... train_size={0}, test_size={1}".format(train_size, len(test_y)))

save_epochs = np.arange(0, n_epoch, save_every)
if (verbose):
    print(" ... saving the following epochs:", save_epochs)
    print(" ... saving the last {0} epochs".format(save_last))

test_losses, train_losses, batch_losses = [], [], []
for epoch in range(n_epoch):
    start_time = time.time()
    epoch_loss = 0.0
    batch_losses = []
    for b in range(n_batch):
        rand_idx = [random.randint(0, train_size) 
                    for _ in range(batch_size)]
        batch = [load_graph(graphs[i]) for i in rand_idx]
        O, Rs, Rr, Ra, y = get_inputs(batch)
        predicted = interaction_network(O, Rs, Rr, Ra)
        batch_loss = criterion(torch.cat(predicted, dim=0), 
                               torch.cat(y, dim=0))
        epoch_loss += batch_loss.item()
        batch_losses.append(batch_loss.item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        print(" ... batch {0}".format(b))
        
    end_time = time.time()
    train_losses.append(epoch_loss/n_batch)
    
    # compute test loss
    test_loss_sum = 0
    for test_graph in graphs[train_size:]:
        O, Rs, Rr, Ra, y = get_inputs([load_graph(test_graph)])
        predicted = interaction_network(O, Rs, Rr, Ra)
        test_loss_sum += criterion(torch.cat(predicted, dim=0), 
                                   torch.cat(y, dim=0)).item()
    
    test_loss = test_loss_sum/test_size
    test_losses.append(test_loss)
        
    print(" * Epoch {0}: time={1:2.2f}s, test_loss={2:0.3f}, train_loss={2:0.3f}"
          .format(epoch, end_time-start_time, test_loss, epoch_loss/n_batch))


    if (epoch in save_epochs) or (epoch  > n_epoch - save_last):
        outfile = "{0}/{1}_epoch{2}.pt".format(model_outdir, job_name, epoch)
        torch.save(interaction_network.state_dict(), outfile)

plotLosses(test_losses, train_losses, "{0}/losses_{1}.png".format(plot_outdir, job_name))

