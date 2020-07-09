# GNNs Algorithms for Tracklet Finding in Pixel Detectors at the HL-LHC

## Overview
This repository implements multiple Graph Neural Network architectures designed to improve the accuracy and efficiency of [particle track finding](https://indico.cern.ch/event/96989/contributions/2124495/attachments/1114189/1589705/WellsTracking.pdf) in pixel detectors (such as those in the [CMS](https://home.fnal.gov/~souvik/CMSPixels/index.html) and [ATLAS](https://www.slac.stanford.edu/econf/C020909/mgspaper.pdf) experiments) for the [High Luminosity LHC](https://home.cern/science/accelerators/high-luminosity-lhc). For more information on tracking at the HL-LHC please see [1] - [3]. This project is supported by the [Insititue for Research and Innovation in Software for High Energy Physics (IRIS-HEP)](https://iris-hep.org/) and maintained by [Savannah Thais](https://github.com/savvy379) and [Gage DeZoort](https://github.com/GageDeZoort). We work closely with the [ExaTrkX project](https://github.com/exatrkx) and Mark Neubauer's [group](https://github.com/Neubauer-Group) at UIUC. 

## Quickstart
This code uses data from the CERN [TrackML Challenge](https://www.kaggle.com/c/trackml-particle-identification/overview) which can be downloaded from Kaggle. This repository includes the python library to easily read the data into a dataframe in the `trackml-library` folder. Upon first downloading this repository you must install this library by running 
``` 
pip install /path/to/repository
```

 To run a GNN model over the Track ML dataset simply run 
 ```
 python prepare_XX.py configs/XX/prepare/config_name.yaml
 ```
 to create the graphs and
 ```
 python train_XX.py configs/XX/train/config_name.yaml
```
to train the GNN. 

In both commands, 'XX' should be replaced with the name of your chosen model. There are many example config files for running with different configurations (i.e. including endcaps, data augmentation, etc) but you can also create a custom config for your task. Be sure to change the `input directory` and `output directory`commands in the config files to match your individual system. 

**Note:** the prepare.py and train.py scripts must be run from the main directory. We include the core scripts for both models (Edge Classifier 1 and Interaction Network) there; additional prepare and train scripts can be found in the script folder. 

## Dataset and Graph Construction

### TrackML Data
All example code in this repository uses the CERN [TrackML Challenge](https://www.kaggle.com/c/trackml-particle-identification/overview) dataset. Each event is separated into four files: cells, hits, particles, and truth; we rely on hits to build the graphs and particles + truth to define the training labels. 

The data is simulated at high pileup using a toy detector that roughly approximates the ATLAS detector. It is divided into modules (detector components) each of which is then divided into several layers as shown below. Here, we focus on the inner-most pixel layers, but this can be adjusted in the prepare.py scripts. 

![Detector layout](https://asalzbur.web.cern.ch/asalzbur/work/tml/Detector.png)

### Data Preprocessing
Each TrackML event is converted into a directed multigraph of hits connected by segments. Different pre-processing strategies are available, each with different graph construction efficiencies. This repo contains two such stratigies:
   1) Select one hit per particle per layer, connect hits in adjacent layers. This is the strategy used by the [HEP.TrkX collaboration](https://heptrkx.github.io/), which we denote "layer pairs" (see **prep_LP.py**) [2]. 
   2) Select hits between adjacent layers *and* hits within the same layer, requiring that same-layer hits are within some distance dR of each other (see **prep_LPP.py**).

## GNNs and Training
The models folder contains the structures to create graphs and multiple GNN models.   

**model/graph.py**: defines a graph as a namedTuple containing:  
* **x**:   a size (N<sub>hits</sub> x 3) feature vector for each hit, containing the hit coordinates (r, phi, z). 
* **R_i**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is incoming to hit h, and 0 otherwise. 
* **R_o**: a size (N<sub>hits</sub> x N<sub>segs</sub>) matrix whose entries (**R_i**)<sub>hs</sub> are 1 when segment s is outgoing from hit h, and 0 otherwise. 
* **a (only for Interaction Network)**:   a size (N<sub>segs</sub> x 1) vector whose s<sup>th</sup> entry is 0 if the segment s connects opposite-layer hits and 1 if segment s connects same-layer hits. 

### Models

#### Edge Classifier
This model is based on the 2018 ExaTrkX [paper](https://arxiv.org/abs/1810.06111) which imploys a set of recurrent iterations of alternation EdgeNetwork and NodeNetwork components:
* **NodeNetwork:** computes new features for every node using the edge weight aggregated features of the connected nodes on the previous and next detector layers separately as well as the nodes' current features (a traditional [Graph Convolutional Network](https://tkipf.github.io/graph-convolutional-networks/)
* **Edge Network:** computes weights for every edge of the graph using the features of the start and end nodes.

#### Interaction Network

### Training
The scripts **train_XX.py** build and train the specified GNN on preprocessed TrackML graphs. The training definitions for each model live in the **models/trainers** folder.

The training data is organized into mini-batches, which are used to optimize the loss fuction specified in your config file using your selected optimizer. The network is then tested on a separate sample of pre-processed graphs. The results of the trained network can be tested using various scripts described in the **Analyses** Section below. 

### Slurm Scripts

## Analyses

### Graph Construction Metrics

### Model Metrics

### Visualizations

### Track Finding

## References
[1] "HL-LHC Tracking Challenge", [talk](https://cds.cern.ch/record/2312314?ln=en) by Jean-Roch Vlimant  
[2] "Tracking performance with the HL-LHC ATLAS detector", [paper](https://cds.cern.ch/record/2683174) by the ATLAS Collaboration  
[3] "Expected Performance of Tracking in CMS at the HL-LHC", [paper](https://www.epj-conferences.org/articles/epjconf/abs/2017/19/epjconf_ctdw2017_00001/epjconf_ctdw2017_00001.html) by the CMS Collaboration
