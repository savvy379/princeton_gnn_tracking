import os
import random
import time
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

from models.IN.interaction_network import InteractionNetwork
from models.IN.graph import Graph, save_graphs, load_graph
import analyses.training_plots as plots

# note that this script is configured by configs/train_IN.yaml
# >>> python plot_IN.py configs/train_IN.yaml
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_IN.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/train_IN.yaml')
    return parser.parse_args()
    
# grab config parameters
args = parse_args()
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)

prep        = config['prep']
pt_cut      = config['pt_cut']
n_epoch     = config['n_epoch']
batch_size  = config['batch_size']
model_dir   = config['model_outdir']
save_last   = config['save_last_n_epoch']
tag         = config['tag']
phi_reflect = config['phi_reflect']
graph_indir   = config['graph_indir']
train_size, test_size = 800, 200
plot_dir = "/tigress/jdezoort/IN_output/plots"

if (phi_reflect): train_size = 1600

# job name, ex. "LP_0p5_1600_noPhi"
job_name = "{0}_{1}_{2}_{3}".format(prep, pt_cut, n_epoch, tag)
print("job_name={0}".format(job_name))

# get model paths
models = os.listdir(model_dir)
models = ["{0}/{1}".format(model_dir, model) 
          for model in models if job_name in model]

# load in test graph paths
graph_dir = "{0}/IN_{1}_{2}/".format(graph_indir, prep, pt_cut)
graph_files = []
if (phi_reflect):
    phi_graph_dir = "/tigress/jdezoort/IN_samples_large/IN_{0}_{1}_phi_reflect/".format(prep, pt_cut)
    graph_files += os.listdir(phi_graph_dir)[0:800]

graph_files += os.listdir(graph_dir)
print("len(graph_files)={0}".format(len(graph_files)))
test_graphs = np.array([(int(f.split('_')[0].split('t00000')[1]),
                         load_graph(graph_dir+f)) for f in graph_files[train_size:]])

# prepare test graphs
size = len(test_graphs)
print("size={0}".format(size))
test_O  = [Variable(torch.FloatTensor(test_graphs[i][1].X))
           for i in range(size)]
test_Rs = [Variable(torch.FloatTensor(test_graphs[i][1].Ro))
           for i in range(size)]
test_Rr = [Variable(torch.FloatTensor(test_graphs[i][1].Ri))
           for i in range(size)]
test_Ra = [Variable(torch.FloatTensor(test_graphs[i][1].a)).unsqueeze(0)
           for i in range(size)]
test_y  = [Variable(torch.FloatTensor(test_graphs[i][1].y)).unsqueeze(0).t()
           for i in range(size)]

# grab the IN with the best test performance
criterion = nn.BCELoss()
interaction_network = InteractionNetwork(3, 1, 1)
losses, epochs = [], []
for i, model in enumerate(models):
    epochs.append(int(model.split('epoch')[1].split('.')[0]))
    interaction_network.load_state_dict(torch.load(model))
    interaction_network.eval()
    predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)
    predicted = torch.cat(predicted, dim=0)
    target = torch.cat(test_y, dim=0)
    loss = (criterion(predicted, target))
    losses.append(loss.item())
    print("Model {0} (Epoch #{1}):".format(model, n_epoch-i), loss.item())

plots.plot_losses(epochs, losses, "{0}/{1}_test_losses.png".format(plot_dir, job_name))
  
best_idx = np.argsort(np.array(losses))[0]   
print("\nBest Loss:", losses[best_idx])
print("Selected model {0}".format(models[best_idx]))
interaction_network.load_state_dict(torch.load(models[best_idx]))
interaction_network.eval()
predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)

losses = [criterion(predicted[i], test_y[i]) for i in range(size)]

# shape up data for analysis
out = predicted
predicted = torch.cat(predicted, dim=0)
to_block_diagram = predicted
predicted = np.array([float(predicted[i].data[0])
                      for i in range(len(predicted))])
target = torch.cat(test_y, dim=0)
target = np.array([float(target[i].data[0])
                   for i in range(len(target))])

real_seg_idx = (target==1).nonzero()[:][0]
real_seg = predicted[real_seg_idx]
fake_seg_idx = (target==0).nonzero()[:][0]
fake_seg = predicted[fake_seg_idx]

# order some plots
plots.plotDiscriminant(real_seg, fake_seg, 20, "{0}/{1}_discriminant.png".format(plot_dir, job_name))

for i in np.arange(0, 1, 0.01):
    testConfusion = plots.confusionMatrix(real_seg, fake_seg, i)
    print(i, testConfusion)

sort_idx = np.argsort(losses)

plots.confusionPlot(real_seg, fake_seg, "{0}/{1}_confusions.png".format(plot_dir, job_name))
plots.plotROC(real_seg, fake_seg, "{0}/{1}_ROC.png".format(plot_dir, job_name))

for num, i in enumerate(sort_idx[-1:]):
    print("#{0}, loss={1}".format(num, losses[i]))
    plots.draw_graph_rz(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_rz_worst{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_rz(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_rz_worst{2}_showErrors.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_xy_worst{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_xy_worst{2}_showErrors.png".format(plot_dir, job_name, num),
                        highlight_errors=True)
    
for num, i in enumerate(sort_idx[:1]):
    print("#{0}, loss={1}".format(num, losses[i]))
    plots.draw_graph_rz(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_rz_best{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_rz(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_rz_best{2}_showErrors.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_xy_best{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(test_O[i], test_Rr[i], test_Rs[i], test_y[i].squeeze(), out[i].squeeze(),
                        filename="{0}/{1}_xy_best{2}_showErrors.png".format(plot_dir, job_name, num),
                        highlight_errors=True)

