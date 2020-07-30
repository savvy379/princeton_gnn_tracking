import os
import random
import time
import argparse

import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.IN.interaction_network import InteractionNetwork
from models.IN.graph import Graph, save_graphs, load_graph
import analyses.prepare_plots as plots
import analyses.track_finding_tools as tf_tools

# note that this script is configured by configs/train_IN.yaml
# >>> python track_finding_IN.py configs/IN/train_IN.yaml
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
tag         = config['tag']
graph_indir = config['graph_indir']
train_size, test_size = 800, 200
plot_dir = "/tigress/jdezoort/IN_output/plots"

# job name, ex. "LP_0p5_1600_noPhi"
job_name = "{0}_{1}_{2}_{3}".format(prep, pt_cut, n_epoch, tag)
print("job_name={0}".format(job_name))

# load in test graph paths
graph_dir = "{0}/IN_{1}_{2}/".format(graph_indir, prep, pt_cut)
graph_files = os.listdir(graph_dir)

# (<evt_id>, <Graph>), e.g. evt_id=1582
test_graphs = np.array([(int(f.split('_')[0].split('t00000')[1]),
                         load_graph(graph_dir+f)) 
                        for f in graph_files[train_size:]])

# truth info, particle_id for each hit
particle_ids = [pickle.load(open("truth/"+str(test_graphs[i][0])+"truth.pkl", 'rb'))
                for i in range(test_size)]

# load in test_graphs
test_O  = [Variable(torch.FloatTensor(test_graphs[i][1].X))
           for i in range(test_size)]
test_Rs = [Variable(torch.FloatTensor(test_graphs[i][1].Ro))
           for i in range(test_size)]
test_Rr = [Variable(torch.FloatTensor(test_graphs[i][1].Ri))
           for i in range(test_size)]
test_Ra = [Variable(torch.FloatTensor(test_graphs[i][1].a)).unsqueeze(0)
           for i in range(test_size)]
test_y  = [Variable(torch.FloatTensor(test_graphs[i][1].y)).unsqueeze(0).t()
           for i in range(test_size)]

# load up a good IN
model_dir = "/tigress/jdezoort/IN_output/trained_models/"
model = model_dir + "LP_2_350_thin_LR03_recur1_endcaps_epoch345.pt"

criterion = torch.nn.BCELoss()
interaction_network = InteractionNetwork(3, 1, 1)
interaction_network.load_state_dict(torch.load(model))
interaction_network.eval()    
predicted = interaction_network(test_O, test_Rs, test_Rr, test_Ra)
losses = [criterion(predicted[i], test_y[i]).item()
          for i in range(test_size)] 

# to hist
eps_range = [0.20]
bad_fractions = {eps:[] for eps in eps_range}
IDd_fractions = {eps:[] for eps in eps_range}
times = []

# run DBScan on each test graph
for g in range(test_size):

    # grab the graph's matrices
    X = test_O[g].detach().numpy()
    Ri = test_Rr[g].detach().numpy()
    Ro = test_Rs[g].detach().numpy()
    y  = test_y[g].detach().numpy()

    # add hit index to features, re-scale phi
    X = np.array([[i, X[i][0], X[i][1]*np.pi, X[i][2]]
                  for i in range(len(X))])

    # build array representing good/bad edges 
    edge_weights = predicted[g].detach().numpy()
    WP = 0.54
    true_segs = np.array(edge_weights >= WP, dtype=bool).flatten()
    
    # map features onto segments, keep only weight=1 segs
    feats_i = np.matmul(Ri.transpose(), X)
    feats_o = np.matmul(Ro.transpose(), X)
    pred_i = np.matmul(Ri.transpose(), X)[true_segs]
    pred_o = np.matmul(Ro.transpose(), X)[true_segs]
    
    # calculate n_segs per hit
    n_segs = ([np.sum(Ri[h]) + np.sum(Ro[h]) for h in range(X.shape[0])])
    
    # calculate n_good_segs per hit
    n_good_segs = ([np.sum(Ri[h][true_segs]) + np.sum(Ro[h][true_segs])
                    for h in range(X.shape[0])])
    
    # calculate distances between connected points
    r_i, phi_i, z_i = pred_i[:, 1], pred_i[:, 2], pred_i[:, 3]
    r_o, phi_o, z_o = pred_o[:, 1], pred_o[:, 2], pred_o[:, 3]
    distances = np.sqrt((r_i*np.cos(phi_i) - r_o*np.cos(phi_o))**2 + 
                        (r_i*np.sin(phi_i) - r_o*np.sin(phi_o))**2 +
                        (z_i-z_o)**2, dtype=float)
    
    # FOR DEBUGGING: define a hit_i, hit_j -> dist(i->j) dictionary
    dist_dict = {tuple([int(pred_i[j][0]), int(pred_o[j][0])]):distances[j] 
                     for j in range(len(distances))}
    
    # (dist_matrix)_ij = dist(i->j) for segment i->j 
    # note that this matrix is non-symmetric
    dist_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for h in range(len(pred_i)):
        dist_matrix[int(pred_o[h][0])][int(pred_i[h][0])] = distances[h]

    # DBScan
    eps, min_pts = 0.2, 1
    hit_clusters, time = tf_tools.DBScan(dist_matrix, eps=eps, min_pts=min_pts)
    times.append(time)
    n_clusters = int(np.max(hit_clusters))
        
    # validate
    hit_particle_ids = particle_ids[g]
    false_counts, true_counts = 0, 0
    for i in range(n_clusters):
        clustered_hits_idx = np.where(hit_clusters==i)[0]
        clustered_hits = X[clustered_hits_idx]
        clustered_particles = hit_particle_ids[clustered_hits_idx]
        if (np.unique(clustered_particles).shape[0] > 1):
            if (i != 0): false_counts += 1
        elif (np.unique(clustered_particles).shape[0] == 1):
            if (i != 0): 
                true_counts += 1
    
    n_tracks = np.unique(hit_particle_ids).shape[0]
    print("ID'd {0}/{1} tracks".format(true_counts, n_tracks))
    IDd_fraction = true_counts/float(n_tracks)
    IDd_fractions[eps].append(IDd_fraction)
    
    bad_fraction = false_counts/float(n_clusters)
    print("Bad fraction: {0}".format(bad_fraction))
    bad_fractions[eps].append(bad_fraction)

    #plot_N = 20
    #filename = "{0}/DBScan_{1}_{2}".format(plot_dir, plot_N, test_graphs[g][0])
    #tf_tools.plot_N_clusters(plot_N, X, feats_i, feats_o, hit_clusters, filename)

plots.plotSingleHist(times, "Time [s]", "Counts", 16, color="mediumslateblue")

for eps, frac_list in bad_fractions.items():
    plots.plotSingleHist(frac_list, "Bad Fraction", "Counts", 16, 
                         weights=None, title="$\epsilon=0.2$, minpts$=1$", 
                         color='red')
    plots.plotXY(losses, frac_list, "Loss", "Bad Fraction")
    print("eps: {0} -> {1} +/- {2}"
          .format(eps, np.mean(frac_list), np.sqrt(np.var(frac_list))))

for eps, frac_list in IDd_fractions.items():
    plots.plotSingleHist(frac_list, "ID'd Fraction", "Counts", 16,
                         weights=None, title="$\epsilon=0.2$, minpts$=1$", 
                         color="mediumseagreen")
    plots.plotXY(losses, frac_list, "Loss", "ID'd Fraction")
    print("eps: {0} -> {1} +/- {2}"
          .format(eps, np.mean(frac_list), np.sqrt(np.var(frac_list))))


#def plotXY(x, y, x_label, y_label, yerr = [], title='', color='blue', save=False):

#def plotXYXY(x1, y1, label_1, x2, y2, label_2, x_label, y_label, yerr1 = [], yerr2 = [],
#             title='', color1='blue', color2 = 'seagreen', save=False):

    # plot_N = 7
    # filename = "{0}/DBScan_{1}_{2}".format(plot_dir, plot_N, test_graphs[g][0]) 
    # tf_tools.plot_N_clusters(7, X, feats_i, feats_o, hit_clusters, filename)
