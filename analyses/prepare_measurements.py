#!/usr/bin/env python

import os
import sys
import yaml
import pickle
import argparse
sys.path.append("../")

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import poisson

import prepare_plots as plots

parser = argparse.ArgumentParser('prepare_measurements.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='prepare_measurements.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)

model    = config['model']
strategy = config['strategy']

if (model == 'IN'): 
    from models.IN.graph import Graph, load_graph
elif ('EC' in model):
    from models.EC1.graph import Graph, load_graph

n_events   = config['n_events']
pt_cuts    = config['pt_cuts']
graph_dir  = config['graph_dir']
truth_file = config['truth_file']

print("analyzing\n  - {0}\n  - {1}_{2}\n  - {3}"
      .format(pt_cuts, model, strategy, truth_file))

# configure matplotlib
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)
rc('text', usetex=True)

def get_shape(g):
    """ graph shape = (n_nodes, n_edges)
    """
    n_nodes = g[1].X.shape[0]
    n_edges = g[1].Ri.shape[1]
    return n_nodes, n_edges

def get_segment_efficiency(g):
    """ segment efficiency = sum(y)/len(y)
    """
    print(" ...sum(y)={0}, len(y)={1}"
          .format(np.sum(g[1].y), g[1].y.shape[0]))
    return np.sum(g[1].y)/(g[1].y).shape[0]

def get_truth_efficiency(i, g, tr):
    """ truth efficiency = sum(y)/n_total_true
    """
    evt_id = g[0]
    n_segs = tr[evt_id][0]
    pt_cuts = tr['pt_cuts'][i]
    truth_eff = np.sum(g[1].y)/n_segs[i]
    print(" ...sum(y)={0}, n_truth={1}"
          .format(np.sum(g[1].y), n_segs[i]))
    #print(np.quantile(truth_effs, 0.5, axis=0))
    return truth_eff

def read_truth(file_name):
    with open(file_name, 'rb') as f:
        truth_table = pickle.load(f)
    return truth_table


truth_table = read_truth(truth_file)
data_dirs  = ["{0}/{1}_{2}_{3}/".format(graph_dir, model, strategy, pt)
              for pt in pt_cuts]

to_hist = df({'seg_eff'   : [], 'seg_eff_er'   : [],
              'truth_eff' : [], 'truth_eff_er' : [],
              'part_eff'  : [], 'part_eff_er'  : [],
              'n_segs'    : [], 'n_segs_er'    : [],
              'n_nodes'   : [], 'n_nodes_er'   : [],
              'size'      : [], 'size_er'      : []})

for i in range(len(pt_cuts)):
    print(" pt: {0}".format(pt_cuts[i]))
    
    graph_files = os.listdir(data_dirs[i])
    seg_effs, truth_effs = [], []
    node_counts, edge_counts = [], []
    sizes = []

    for graph_file in graph_files:
        graph_path = data_dirs[i] + graph_file
        graph_data = (graph_file.split('_')[0],
                      load_graph(graph_path),
                      os.path.getsize(graph_path)/10**6)
        sizes.append(graph_data[2])
        print("... analyzing {0}".format(graph_data[0]))
        print("... truth: {0}".format(truth_table[graph_data[0]]))

        n_nodes, n_edges = get_shape(graph_data)
        node_counts.append(n_nodes)
        edge_counts.append(n_edges)
        
        truth_eff = get_truth_efficiency(i, graph_data, truth_table)
        seg_eff = get_segment_efficiency(graph_data)
        print("... truth_eff: {0}".format(truth_eff))
        print("... seg_eff: {0}".format(seg_eff))
  
        truth_effs.append(truth_eff)
        seg_effs.append(seg_eff)
        
    avg_seg_eff   = [np.mean(seg_effs),
                     np.quantile(seg_effs, 0.25, axis=0),
                     np.quantile(seg_effs, 0.75, axis=0)]
    avg_truth_eff = [np.mean(truth_effs),
                     np.quantile(truth_effs, 0.25, axis=0), 
                     np.quantile(truth_effs, 0.75, axis=0)]
    
    avg_nodes     = [np.mean(node_counts), np.sqrt(np.var(node_counts))]
    avg_edges     = [np.mean(edge_counts), np.sqrt(np.var(edge_counts))]
    avg_size      = [np.mean(sizes), np.sqrt(np.var(sizes))]

    data_tag = " ***** pt=" + pt_cuts[i] + " data ***** "
    print("{0}\n \t seg_eff: {1} +/- {2} \n \t truth_eff: {3} +/- {4}"
          .format(data_tag, np.round(avg_seg_eff[0], decimals=3),
                  np.round(avg_seg_eff[1], decimals=3),
                  np.round(avg_truth_eff[0], decimals=3),
                  np.round(avg_truth_eff[1], decimals=3)))
    print("\t nodes: {0} +/- {1} \n \t edges: {2} +/- {3}"
          .format(np.round(avg_nodes[0], decimals=3),
                  np.round(avg_nodes[1], decimals=3),
                  np.round(avg_edges[0], decimals=3),
                  np.round(avg_edges[1], decimals=3)))
    print("\t size: {0} +/- {1}"
          .format(np.round(avg_size[0], decimals=3),
                  np.round(avg_size[1], decimals=3)))
    
    to_hist = to_hist.append({'seg_eff'      : avg_seg_eff[0],
                              'seg_eff_up'   : avg_seg_eff[1],
                              'seg_eff_dn'   : avg_seg_eff[2],
                              'truth_eff'    : avg_truth_eff[0],
                              'truth_eff_up' : avg_truth_eff[1],
                              'truth_eff_dn' : avg_truth_eff[2],
                              'n_segs'       : avg_edges[0],
                              'n_segs_er'    : avg_edges[1],
                              'n_nodes'      : avg_nodes[0],
                              'n_nodes_er'   : avg_nodes[1],
                              'size'         : avg_size[0],
                              'size_er'      : avg_size[1]},
                             ignore_index=True)

pt_map = {'0p5':0.5, '0p6':0.6, '0p75':0.75, '1':1.0,
          '1p5':1.5, '2p5':2.5, '3':3, '4':4, '5':5}
pt_cuts = np.array(pt_map[pt_cut] for pt_cut in pt_cuts)

plots.plotQuant(pt_cuts, np.array(to_hist['truth_eff']), 
                np.array(to_hist['truth_eff_up']), 
                np.array(to_hist['truth_eff_dn']),
                '$p_{T}$ Cut [GeV]', 'Truth Efficiency',
                title=graph_dir+'/plots/truth_eff.png', 
                save_fig=True)

plots.plotQuant(pt_cuts, np.array(to_hist['seg_eff']),
                np.array(to_hist['seg_eff_up']),
                np.array(to_hist['seg_eff_dn']),
                '$p_{T}$ Cut [GeV]', 'Segment Efficiency',
                title=graph_dir+'/plots/seg_eff.png',
                save_fig=True)

plotXY(pt_cuts, np.array(to_hist['size']), '$p_{T}$ Cut [GeV]',
       'Size [MB]', yerr=np.array(to_hist['size_er']), color='indigo',
       title=graph_dir+'/plots/size.png', save_Fig=True)

plotXY(pt_cuts, np.array(to_hist['n_segs']), '$p_{T}$ Cut [GeV]',
       'Segments per Graph', yerr=np.array(to_hist['n_segs_er']), color='seagreen',
       title=graph_dir+'/plots/n_segments.png', save_fig=True)

plotXY(pt_cuts, np.array(to_hist['n_nodes']), '$p_{T}$ Cut [GeV]',
       'Nodes per Graph', yerr=np.array(to_hist['n_nodes_er']), color='seagreen',
       title=graph_dir+'/plots/n_nodes.png', save_fig=True)



"""
plotXY(pt_cuts, np.array(build_time)/float(n_events),
       '$p_{T}$ Cut [GeV]', 'Avg. Construction Time [s]', color='indigo',
       title='construction_time')
plotXY(pt_cuts[4:], np.array(build_time)[4:]/float(n_events),
       '$p_{T}$ Cut [GeV]', 'Avg. Construction Time [s]', color='indigo',
       title='construction_time_zoomed')
plotXY(pt_cuts, np.array(to_hist['size']), '$p_{T}$ Cut [GeV]', 
       'Size [MB]', yerr=np.array(to_hist['size_er']), color='indigo',
       title='size')
plotXY(pt_cuts[4:], np.array(to_hist['size'])[4:], '$p_{T}$ Cut [GeV]',
       'Size [MB]', yerr=np.array(to_hist['size_er'])[4:], color='indigo',
       title='size_zoomed')
plotXY(pt_cuts, np.array(to_hist['seg_eff']), '$p_{T}$ Cut [GeV]', 
       'True/Total Selected Segments', yerr=np.array(to_hist['seg_eff_er']), color='dodgerblue',
       title='segment_efficiency')
plotXY(pt_cuts, np.array(to_hist['truth_eff']), '$p_{T}$ Cut [GeV]', 
       'Selected/Total True Segments', yerr=np.array(to_hist['truth_eff_er']), color='dodgerblue',
       title='truth_efficiency')
plotXY(pt_cuts, np.array(to_hist['n_segs']), '$p_{T}$ Cut [GeV]', 
       'Segments per Graph', yerr=np.array(to_hist['n_segs_er']), color='seagreen',
       title='n_segments')
plotXY(pt_cuts, np.array(to_hist['n_nodes']), '$p_{T}$ Cut [GeV]',
       'Nodes per Graph', yerr=np.array(to_hist['n_nodes_er']), color='seagreen',
       title='n_nodes')
"""
