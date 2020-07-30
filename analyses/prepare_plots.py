#!/usr/bin/env python

""" plot_functions.py: suite of visualizations and kinematic studies for TrackML events """ 

__author__  = "Gage DeZoort"
__version__ = "1.0.0"
__status__  = "Development"

import os
import math

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset


# configure matplotlib
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('text', usetex=True)


def plot_rz(X, Ri, Ro):
    X = np.array(X)
    X[:, 0] *= np.sign(X[:, 1])
    feats_o = np.matmul(Ro.transpose(), X)
    feats_i = np.matmul(Ri.transpose(), X)

    for i in range(len(X)):        
        plt.scatter(X[i][2], X[i][0], c='silver', linewidths=0, marker='s', s=8)

    for i in range(len(feats_o)):
        plt.plot((feats_o[i][2], feats_i[i][2]),
                 (feats_o[i][0], feats_i[i][0]),
                 'bo-', lw=0.1, ms=0.1, alpha=0.3)

    plt.ylabel("R [m]")
    plt.xlabel("z [m]")
    #plt.savefig(filename, dpi=1200)
    plt.show()
    plt.clf()


def plotSingleHist(data, x_label, y_label, bins, weights=None, title='', color='blue'):
    """ plotSingleHist(): generic function for histogramming a data array
    """
    bin_heights, bin_borders, _ = plt.hist(data, bins=bins, color=color, weights=weights)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2.
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title,    fontsize=16)
    plt.show()

def plotDoubleHistOverlapped(data1, data2, xlabel, ylabel, nBins, minBin, maxBin,
                             label1="", label2="", color1='blue', color2='red', 
                             figLabel="", saveFig=False):
    bins = np.linspace(minBin, maxBin, num=nBins)
    bin_heights1, bin_borders1, _ = plt.hist(data1, bins = bins, alpha=0.5,
                                             color = color1, label=label1) #, histtype='step')
    bin_centers1 = bin_borders1[:-1] + np.diff(bin_borders1)/2.
    bin_heights2, bin_borders2, _ = plt.hist(data2, bins = bins, alpha=0.5,
                                             color = color2, label=label2) #,  histtype='step')
    bin_centers2 = bin_borders2[:-1] + np.diff(bin_borders2)/2.
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='upper right')
    if saveFig:
        plt.savefig(figLabel, dpi=1200)
    plt.show()

def plotQuant(x, y, up, down, xlabel, ylabel, title='', 
              color='mediumseagreen', save_fig=False):
    print(x, y, up, down)
    plt.errorbar(x, y, yerr=np.array([y-down, up-y]), color=color, capsize=1, ls='', lw=1, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (save_fig): plt.savefig(title+'.png', dpi=1200)
    plt.show()

def plotXY(x, y, x_label, y_label, yerr = [], title='', color='blue', save_fig=False):
    """ generic function for scatter plotting
    """
    if (len(yerr) == 0): plt.scatter(x, y, c=color)
    else: plt.errorbar(x, y, yerr=yerr, c=color, capsize=1, ls='', lw=1, marker='.')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if (save_fig): plt.savefig(title+'.png', dpi=1200)
    plt.show()

def plotXYXY(x1, y1, label_1, x2, y2, label_2, x_label, y_label, yerr1 = [], yerr2 = [],
             title='', color1='blue', color2 = 'seagreen', save=False):
    """ generic function for scatter plotting two scatters
    """
    plt.errorbar(x1, y1, label=label_1, yerr=yerr1, c=color1, capsize=1, ls='', lw=1, marker='.')
    plt.errorbar(x2, y2, label=label_2, yerr=yerr2, c=color2, capsize=1, ls='', lw=1, marker='.')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper left")
    if (save):
        plt.savefig(title+'.png', dpi=1200)
    plt.show()
    
def plotBinnedHist(x, y, x_label, y_label, nbins, title='', color='blue'):
    """ plotBinnedHist(): generic function for plotting a binned y vs. x scatter plot
    					  similar to a TProfile
    """
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)/np.sqrt(n)
    plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, color=color, marker='.', ls='')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def plotHeatMap(x, y, x_label, y_label, x_bins, y_bins, weights=None, title=''):
    """ plotHeatMap(): generic function for plotting a heatmap of y vs. x with 
                       pre-specified weights
    """
    plt.hist2d(x, y, bins=(x_bins,y_bins), weights=weights, cmap=plt.cm.jet)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normalized $P_T$')
    plt.title(title)
    plt.show()

def plotTrack(track):
    """ plotTrack(): plot a track in a simple 3D grid 
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(track['tx'], track['ty'], track['tz'], lw=0.5, c='skyblue')
    ax.scatter3D(track['tx'], track['ty'], track['tz'], 
                 c=track['tR'], cmap ='viridis', marker='h', s=30)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    plt.show()
    
def getModuleCoords(v_id, l_id, m_id):
    detectors = pd.read_csv('../data/detectors.csv')
    coords = detectors[(detectors['volume_id'] == v_id) & (detectors['layer_id'] == l_id) 
                       & (detectors['module_id'] == m_id)]
    
    c_vec = [coords.iloc[0]['cx'], coords.iloc[0]['cy'], coords.iloc[0]['cz']]
    hu = coords.iloc[0]['module_maxhu']
    hv = coords.iloc[0]['module_hv']
    
    def rotateCoords(vec):
        rotation_matrix = np.array([[coords.iloc[0]['rot_xu'],coords.iloc[0]['rot_xv'],coords.iloc[0]['rot_xw']],
                           [coords.iloc[0]['rot_yu'],coords.iloc[0]['rot_yv'],coords.iloc[0]['rot_yw']],
                           [coords.iloc[0]['rot_zu'],coords.iloc[0]['rot_zv'],coords.iloc[0]['rot_zw']]])
        return rotation_matrix.dot(vec)


    v1 = rotateCoords(np.array([-hu,-hv,0]))
    v2 = rotateCoords(np.array([hu,-hv,0]))
    v3 = rotateCoords(np.array([hu,hv,0]))
    v4 = rotateCoords(np.array([-hu,hv,0]))
    
    x = np.array([v1[0],v2[0],v3[0],v4[0]]) + c_vec[0]
    y = np.array([v1[1],v2[1],v3[1],v4[1]]) + c_vec[1]
    z = np.array([v1[2],v2[2],v3[2],v4[2]]) + c_vec[2]
    
    verts = [list(zip(x,y,z))]
    return verts

def plotTrackOverLayers(track, hits, plotModules):
    """ plotTrackOverLayers(): plot a track and the detector layers
                               it hits
    """
    
    volume_ids = [7,8,9]
    detectors = pd.read_csv('../data/detectors.csv')
    detectors['xyz'] = detectors[['cx', 'cy', 'cz']].values.tolist()
    
    volumes = detectors.groupby('volume_id')['xyz'].apply(list).to_frame()	
    accept_volumes = detectors[detectors.volume_id.isin(volume_ids)]
    
    x_min, x_max = accept_volumes['cx'].min(), accept_volumes['cx'].max()
    y_min, y_max = accept_volumes['cy'].min(), accept_volumes['cy'].max()
    z_min, z_max = accept_volumes['cz'].min(), accept_volumes['cz'].max()
    
    volumes_layers = accept_volumes.groupby(['volume_id','layer_id'])['xyz'].apply(list).to_frame()
    fig = plt.figure(figsize=plt.figaspect(0.9))
    
    ax = plt.axes(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    #ax.set_prop_cycle(cycler('color', ['forestgreen', 'firebrick', 'royalblue', 'indigo']))
    
    ax.plot(track['tx'], track['ty'], track['tz'], lw=0.5, c='skyblue')
    ax.scatter3D(track['tx'], track['ty'], track['tz'],
                 c=track['tR'], cmap='viridis', marker='h', s=30)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    
    if plotModules:
        for index, row in hits.iterrows():
            verts = getModuleCoords(row['volume_id'], row['layer_id'], row['module_id'])
            ax.add_collection3d(Poly3DCollection(verts, facecolors='silver', linewidths=1, edgecolors='black'), zs='z')
    
    num_regions = volumes_layers.shape[0]
    for (i, row) in volumes_layers.iloc[:num_regions+1].iterrows():
        xyz = np.array(row['xyz'])
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        ax.plot(x,y,z, linestyle='', marker='h', markersize='0.5', color='mediumslateblue', alpha=0.5)
        ax.text(x[0], y[0], z[0], str(i), None, size=5)
    plt.show()



