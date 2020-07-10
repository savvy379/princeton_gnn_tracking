#!/usr/bin/env

import os
import sys
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

# break down the input files into event_ids
data_dir = "/home/sthais/data/sample"
files = [f for f in listdir(data_dir)]
files = [f.split('.')[0] for f in files if "hits" in f]
evt_ids = [f.split('-')[0] for f in files]

outfile = open("truth_LPP.txt", "w")

for evt_id in evt_ids:
    
    n_segs_per_graph = []
    pt_cuts = [0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    
    for i in range(len(pt_cuts)):
        
        if (verbose): print(str(evt_id), str(pt_cuts[i]))
        
        # TO-DO: full pixel detector 
        # pixel_layers = [(7,2), (7,4), (7,6), (7,8), (7,10), (7,12), (7,14),
        #                 (8,2), (8,4), (8,6), (8,8),
        #                 (9,2), (9,4), (9,6), (9,8), (9,10), (9,12), (9,14)]
        
        # pixel barrel layers
        pixel_layers = [(8,2), (8,4), (8,6), (8,8)]
        
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
        
        segs_per_particle = []
        particle_ids = hits.particle_id.unique()
        for p in particle_ids:
            particle_hits = hits[hits.particle_id==p]
            n_in_layer = [particle_hits[particle_hits.layer==i].shape[0] for i in range(4)]
            n_segs = 0
            for i in range(len(pixel_layers)):
                if n_in_layer[i] > 1:
                    n_segs += 2*(n_in_layer[i]-1)
                if (i < 3):
                    n_segs += (n_in_layer[i]*n_in_layer[i+1])
            
            segs_per_particle.append(n_segs)
        
        n_segs_per_graph.append(np.sum(segs_per_particle))
        
    outfile.write(str(evt_id) +" "+ str(n_segs_per_graph[0]) +" "+ str(n_segs_per_graph[1])  +" "+  
                  str(n_segs_per_graph[2]) +" "+ str(n_segs_per_graph[3]) +" "+ str(n_segs_per_graph[4]) +" "+
                  str(n_segs_per_graph[5]) +" "+ str(n_segs_per_graph[6]) +" "+ str(n_segs_per_graph[7]) +" "+
                  str(n_segs_per_graph[8]) +" "+ str(n_segs_per_graph[9]) + "\n")
    print(str(evt_id) +" "+ str(n_segs_per_graph[0]) +" "+ str(n_segs_per_graph[1])  +" "+
                  str(n_segs_per_graph[2]) +" "+ str(n_segs_per_graph[3]) +" "+ str(n_segs_per_graph[4]) +" "+
                  str(n_segs_per_graph[5]) +" "+ str(n_segs_per_graph[6]) +" "+ str(n_segs_per_graph[7]) +" "+
                  str(n_segs_per_graph[8]) +" "+ str(n_segs_per_graph[9]) + "\n")
    
outfile.close()
    
