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

# job name, ex. "LP_0p5_1600_noPhi"
job_name = "{0}_{1}_{2}_{3}".format(prep, pt_cut, n_epoch, tag)
print("job_name={0}".format(job_name))

# get model paths
models = os.listdir(model_dir)
models = ["{0}/{1}".format(model_dir, model) 
          for model in models if job_name in model]

# load in test graph paths
graph_dir = "{0}/IN_{1}_{2}/".format(graph_indir, prep, pt_cut)
graph_files = os.listdir(graph_dir)
print("len(graph_files)={0}".format(len(graph_files)))

test_graphs = np.array([(int(f.split('_')[0].split('t00000')[1]), graph_dir+f) 
                        for f in graph_files[train_size:]])
size = len(test_graphs)

# grab the IN with the best test performance
criterion = nn.BCELoss()
interaction_network = InteractionNetwork(3, 1, 1)
losses, epochs = [], []
for i, model in enumerate(models):
    epoch = int(model.split('epoch')[1].split('.')[0])
    #    if (epoch != 20): continue
    epochs.append(epoch)
    interaction_network.load_state_dict(torch.load(model))
    interaction_network.eval()
    model_loss_sum = 0
    for graph in test_graphs[:,1]:
        graph = load_graph(graph)
        O = [Variable(torch.FloatTensor(graph.X))]
        Rs = [Variable(torch.FloatTensor(graph.Ro))]
        Rr = [Variable(torch.FloatTensor(graph.Ri))]
        Ra = [Variable(torch.FloatTensor(graph.a)).unsqueeze(0)]
        y = [Variable(torch.FloatTensor(graph.y)).unsqueeze(0).t()]
        predicted = interaction_network(O, Rs, Rr, Ra)
        predicted = torch.cat(predicted, dim=0)
        target = torch.cat(y, dim=0)
        loss = (criterion(predicted, target))
        model_loss_sum += loss.item()
    
    model_loss = model_loss_sum/test_size
    losses.append(model_loss)
    print("Model {0} (Epoch #{1}):".format(model, i), model_loss)

plots.plot_losses(epochs, losses, "{0}/{1}_test_losses.png".format(plot_dir, job_name))

best_idx = np.argsort(np.array(losses))[0]   
print("\nBest Loss:", losses[best_idx])
print("Selected model {0}".format(models[best_idx]))
interaction_network.load_state_dict(torch.load(models[best_idx]))
interaction_network.eval()
print("GOT TO LOADING BEST MODEL")

losses, real_segs, fake_segs = [], [], []
for graph_file in test_graphs[:,1]:
    graph = load_graph(graph_file)
    O = [Variable(torch.FloatTensor(graph.X))]
    Rs = [Variable(torch.FloatTensor(graph.Ro))]
    Rr = [Variable(torch.FloatTensor(graph.Ri))]
    Ra = [Variable(torch.FloatTensor(graph.a)).unsqueeze(0)]
    y = [Variable(torch.FloatTensor(graph.y)).unsqueeze(0).t()]
    del graph

    predicted = interaction_network(O, Rs, Rr, Ra)
    predicted = torch.cat(predicted, dim=0)
    target = torch.cat(y, dim=0)
    loss = (criterion(predicted, target))
    losses.append(loss.item())

    predicted = predicted.detach().numpy()
    target = target.detach().numpy()
    real_seg_idx = (target == 1)
    real_seg_in_graph = predicted[real_seg_idx]
    fake_seg_idx = (target == 0)
    fake_seg_in_graph = predicted[fake_seg_idx]

    real_segs.append(real_seg_in_graph)
    fake_segs.append(fake_seg_in_graph)
    print("Completed {0}".format(graph_file))

real_segs = np.concatenate(real_segs).ravel()
fake_segs = np.concatenate(fake_segs).ravel()

# shape up data for analysis
#out = predicted
#predicted = torch.cat(predicted, dim=0)
#to_block_diagram = predicted
#predicted = np.array([float(predicted[i].data[0])
#                      for i in range(len(predicted))])
#target = torch.cat(test_y, dim=0)
#target = np.array([float(target[i].data[0])
#                   for i in range(len(target))])##
#
#real_seg_idx = (target==1).nonzero()[:][0]
#real_seg = predicted[real_seg_idx].detach().numpy()
#fake_seg_idx = (target==0).nonzero()[:][0]
#fake_seg = predicted[fake_seg_idx].detach().numpy()

# order some plots
plots.plotDiscriminant(real_segs, fake_segs, 20, "{0}/{1}_discriminant.png".format(plot_dir, job_name))

for i in np.arange(0, 1, 0.01):
    testConfusion = plots.confusionMatrix(real_segs, fake_segs, i)
    print(i, testConfusion)

sort_idx = np.argsort(losses)

plots.confusionPlot(real_segs, fake_segs, "{0}/{1}_confusions.png".format(plot_dir, job_name))
plots.plotROC(real_segs, fake_segs, "{0}/{1}_ROC.png".format(plot_dir, job_name))



for num, i in enumerate(sort_idx[-1:]):

    graph_file = test_graphs[i][1]
    graph = load_graph(graph_file)
    O = [Variable(torch.FloatTensor(graph.X))]
    Rs = [Variable(torch.FloatTensor(graph.Ro))]
    Rr = [Variable(torch.FloatTensor(graph.Ri))]
    Ra = [Variable(torch.FloatTensor(graph.a)).unsqueeze(0)]
    y = [Variable(torch.FloatTensor(graph.y)).unsqueeze(0).t()]
    out = interaction_network(O, Rs, Rr, Ra)
    out = torch.cat(out, dim=0)

    O, Rs, Rr, Ra, y = O[0], Rs[0], Rr[0], Ra[0], y[0]
    print("#{0}, loss={1}".format(num, losses[i]))
    plots.draw_graph_rz(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_rz_worst{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_rz(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_rz_worst{2}_showErrors.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_xy_worst{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_xy_worst{2}_showErrors.png".format(plot_dir, job_name, num),
                        highlight_errors=True)
    
for num, i in enumerate(sort_idx[:1]):

    graph_file = test_graphs[i][1]
    graph = load_graph(graph_file)
    O = [Variable(torch.FloatTensor(graph.X))]
    Rs = [Variable(torch.FloatTensor(graph.Ro))]
    Rr = [Variable(torch.FloatTensor(graph.Ri))]
    Ra = [Variable(torch.FloatTensor(graph.a)).unsqueeze(0)]
    y = [Variable(torch.FloatTensor(graph.y)).unsqueeze(0).t()]
    out = interaction_network(O, Rs, Rr, Ra)
    out = torch.cat(out, dim=0)

    O, Rs, Rr, Ra, y = O[0], Rs[0], Rr[0], Ra[0], y[0]
    print("#{0}, loss={1}".format(num, losses[i]))
    plots.draw_graph_rz(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_rz_best{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_rz(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_rz_best{2}_showErrors.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_xy_best{2}.png".format(plot_dir, job_name, num))
    plots.draw_graph_xy(O, Rr, Rs, y.squeeze(), out.squeeze(),
                        filename="{0}/{1}_xy_best{2}_showErrors.png".format(plot_dir, job_name, num),
                        highlight_errors=True)

