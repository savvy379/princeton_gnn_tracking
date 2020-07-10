#!/usr/bin/env python

import os
import sys
import yaml
import argparse
sys.path.append("../")

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import poisson

import prep_plot_menu as plots

parser = argparse.ArgumentParser('prep_efficiencies.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='prep_IN_efficiencies.yaml')
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

# configure matplotlib
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)
rc('text', usetex=True)

def get_graphs(d):
    """ get all graphs from a directory
          return [("event<id1>", graph1, size), ...]
    """
    files = os.listdir(d)
    return [(f.split('_')[0], load_graph(d+f), os.path.getsize(d+f)/10**6) 
            for f in files]

def get_shape(g):
    """ graph shape = (n_nodes, n_edges)
    """
    n_nodes = [g[i][1].X.shape[0] for i in range(n_events)]
    n_edges = [g[i][1].Ri.shape[1] for i in range(n_events)]
    return np.array([n_nodes, n_edges])

def get_segment_efficiency(g):
    """ segment efficiency = sum(y)/len(y)
    """
    return [np.sum(g[i][1].y)/(g[i][1].y).shape[0]
            for i in range(n_events)]

def get_truth_efficiency(i, gr, tr):
    """ truth efficiency = sum(y)/n_total_true
    """
    print("Truth:", tr.loc[tr['evt_id']==gr[0][0], pt_cuts[i]])
    truth_effs = [np.sum(g[1].y)/float(tr.loc[tr['evt_id']==g[0], pt_cuts[i]]) 
                  for g in gr]
    print(np.quantile(truth_effs, 0.5, axis=0))
    return truth_effs

def print_summary(eff, nodes, edges, tag=""):
    print(tag)
    print("  ==> Average n_nodes = ", np.round(nodes[0], decimals=2),
          "+/-", np.round(nodes[1], decimals=2))
    print("  ==> Average n_edges = ", np.round(edges[0], decimals=2),
          "+/-", np.round(edges[1], decimals=2))
    print("  ==> Segment Efficiency", np.round(eff[0], decimals=3),
          "+/-", np.round(eff[1], decimals=3))    

def read_truth(file_name):
    """ truth file contains the true number of segments per
        pt cut for each file
    """
    truth_file = open(file_name, 'r')
    lines = [i.split(" ") for i in truth_file]
    truth_info = df({'evt_id' : [], '0p5' : [], '0p6' :[],
                     '0p75' : [], '1' : [], '1p5' : [], 
                     '2' : [], '2p5' : [], '3' : [], 
                     '4' : [], '5' : []})
    for line in lines:
        truth_info = truth_info.append({'evt_id' : line[0],
                                        '0p5'    : int(line[1]),
                                        '0p6'    : int(line[2]),
                                        '0p75'   : int(line[3]),
                                        '1'      : int(line[4]),
                                        '1p5'    : int(line[5]),
                                        '2'      : int(line[6]),
                                        '2p5'    : int(line[7]),
                                        '3'      : int(line[8]),
                                        '4'      : int(line[9]),
                                        '5'      : int(line[10])},
                                       ignore_index=True)
        
    return truth_info

truth_info = read_truth(truth_file)
data_dirs  = ["{0}/{1}_{2}_{3}/".format(graph_dir, model, strategy, pt)
              for pt in pt_cuts]

to_hist = df({'seg_eff'   : [], 'seg_eff_er'   : [],
              'truth_eff' : [], 'truth_eff_er' : [],
              'n_segs'    : [], 'n_segs_er'    : [],
              'n_nodes'   : [], 'n_nodes_er'   : [],
              'size'      : [],  'size_er'     : []})

for i in range(len(pt_cuts)):
    data_graphs = get_graphs(data_dirs[i])
    data_shapes = get_shape(data_graphs)
    
    truth_effs = get_truth_efficiency(i, data_graphs, truth_info)
    seg_effs   = get_segment_efficiency(data_graphs)
        
    avg_seg_eff   = [np.mean(seg_effs),
                     np.quantile(seg_effs, 0.05, axis=0),
                     np.quantile(seg_effs, 0.95, axis=0)]
    avg_truth_eff = [np.mean(truth_effs),
                     np.quantile(truth_effs, 0.05, axis=0), 
                     np.quantile(truth_effs, 0.95, axis=0)]
    
    avg_nodes     = [np.mean(data_shapes[0]), np.sqrt(np.var(data_shapes[0]))]
    avg_edges     = [np.mean(data_shapes[1]), np.sqrt(np.var(data_shapes[1]))]
    avg_size      = [np.mean([g[2] for g in data_graphs]),
                     np.sqrt(np.var([g[2] for g in data_graphs]))]

    data_tag = " ***** pt=" + pt_cuts[i] + " data ***** "
    print_summary(avg_seg_eff, avg_nodes, avg_edges, tag=data_tag)
    
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
                '$p_{T}$ Cut [GeV]', 'Truth Efficiency')

plots.plotQuant(pt_cuts, np.array(to_hist['seg_eff']),
                np.array(to_hist['seg_eff_up']),
                np.array(to_hist['seg_eff_dn']),
                '$p_{T}$ Cut [GeV]', 'Segment Efficiency')


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
