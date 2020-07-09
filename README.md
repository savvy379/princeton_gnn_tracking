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

### Data Preprocessing

## GNNs and Training

### Models

#### Edge Classifier

#### Interaction Network

### Training

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
