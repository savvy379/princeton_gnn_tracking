#!/usr/bin/env

import os
import sys
import argparse
from os import listdir
from os.path import isfile, join
sys.path.append("..")
import math

import numpy as np
import pandas as pd

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset

verbose = True
parser = argparse.ArgumentParser('generate_truth_LP.py')
add_arg = parser.add_argument
add_arg('path', nargs='?', default='/tigress/jdezoort/train_1')
args = parser.parse_args()
data_dir = args.path

# break down input files into event_ids 
files = [f for f in listdir(data_dir)]
files = [f.split('.')[0] for f in files if "hits" in f]
evt_ids = [f.split('-')[0] for f in files]

endcaps = True
outfile = open("truth_LP_EC.txt", "w")

for evt_id in evt_ids:
    
    n_segs_per_graph = []
    pt_cuts = [0.5, 0.6, 0.75, 1, 1.5, 2]
    
    for i in range(len(pt_cuts)):

        if (verbose): print(str(evt_id), str(pt_cuts[i]))
        
        pixel_layers = [(8,2), (8,4), (8,6), (8,8)]
        if (endcaps):
            pixel_layers.extend([(7,2), (7,4), (7,6), (7,8),
                                 (7,10), (7,12), (7,14),
                                 (9,2), (9,4), (9,6), (9,8),
                                 (9,10), (9,12), (9,14)])

        hits, cells, particles, truth = load_event(os.path.join(data_dir, evt_id))
        hits_by_loc = hits.groupby(['volume_id', 'layer_id'])
        hits = pd.concat([hits_by_loc.get_group(pixel_layers[i]).assign(layer=i)
                          for i in range(len(pixel_layers))])
        
        pt = np.sqrt(particles.px**2 + particles.py**2)
        particles = particles[pt > pt_cuts[i]]
        
        truth = (truth[['hit_id', 'particle_id', 'weight']]
                 .merge(particles[['particle_id']], on='particle_id'))
    
        r = np.sqrt(hits.x**2 + hits.y**2)
        phi = np.arctan2(hits.y, hits.x)
        
        hits = (hits[['hit_id', 'z', 'layer']]
                .assign(r=r, phi=phi)
                .merge(truth[['hit_id', 'particle_id', 'weight']], on='hit_id'))
        hits = hits.loc[
            hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
        ]
        
        particle_ids = hits.particle_id.unique()
        segs_per_particle = [(hits[hits.particle_id == i].shape[0]-1) 
                             for i in particle_ids]
        n_segs = np.sum(segs_per_particle)
        n_segs_per_graph.append(n_segs)
        
    outfile.write("{0} {1} {2} {3} {4} {5} {6}\n"
                  .format(str(evt_id), str(n_segs_per_graph[0]), 
                          str(n_segs_per_graph[1]), str(n_segs_per_graph[2]), 
                          str(n_segs_per_graph[3]), str(n_segs_per_graph[4]), 
                          str(n_segs_per_graph[5])))
    
    if (verbose):
        print("{0} {1} {2} {3} {4} {5} {6}\n"
              .format(str(evt_id), str(n_segs_per_graph[0]),
                      str(n_segs_per_graph[1]), str(n_segs_per_graph[2]),
                      str(n_segs_per_graph[3]), str(n_segs_per_graph[4]),
                      str(n_segs_per_graph[5])))

        
outfile.close()
    
