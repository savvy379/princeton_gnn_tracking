import tqdm
import torch
from torchviz import make_dot
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches  


def plot_losses(epochs, losses, filename):
    plt.scatter(epochs, losses, c='mediumslateblue', linewidths=0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(filename, dpi=1200)
    plt.clf()

def draw_graph_rz(X, Ri, Ro, y, out,                   
                  cut=0.5, filename='plots/rz.png',
                  highlight_errors=False):
    
    # outgoing and incoming segments
    # to features, dim = (N_hits, 3)
    feats_o = torch.matmul(Ro.t(), X)
    feats_i = torch.matmul(Ri.t(), X)

    X = X.detach().numpy()
    Ri = Ri.detach().numpy()
    Ro = Ro.detach().numpy()
    y = y.detach().numpy()
    out = out.detach().numpy()
    N = len(y)
    result = np.array([((y[i] == 1 and out[i] >= cut), # true positive
                        (y[i] == 1 and out[i] <  cut), # false negative
                        (y[i] == 0 and out[i] >  cut), # false positive
                        (y[i] == 0 and out[i] <= cut)) # true negative
                       for i in range(N)])
    
    for i in range(len(X)):
        plt.scatter(X[i][2], X[i][0], c='silver', linewidths=0, marker='s', s=8)

    for i in range(N):
        if (result[i][0]):
            if (highlight_errors):
                plt.plot((feats_o[i][2], feats_i[i][2]), 
                         (feats_o[i][0], feats_i[i][0]), 
                         'go-', lw=0.1, ms=0.1, alpha=0.3)
            else:
                plt.plot((feats_o[i][2], feats_i[i][2]),
                         (feats_o[i][0], feats_i[i][0]),
                         'go-', lw=0.4, ms=0.2, alpha=0.5)
        if (result[i][2]): # false positive
            plt.plot((feats_o[i][2], feats_i[i][2]),
                     (feats_o[i][0], feats_i[i][0]), 'ro-', lw=0.4, ms=0.2, alpha=0.5)
        if (result[i][1]): # false negative
            plt.plot((feats_o[i][2], feats_i[i][2]),
                     (feats_o[i][0], feats_i[i][0]), 'bo-', lw=0.4, ms=0.2, alpha=0.5)
    
    red_patch = mpatches.Patch(color='red', label='False Positive')
    blue_patch = mpatches.Patch(color='blue', label='False Negative')
    green_patch = mpatches.Patch(color='green', label='True Positive')
    plt.legend(handles=[green_patch, blue_patch, red_patch])
    plt.ylabel("R [m]")
    plt.xlabel("z [m]")
    plt.savefig(filename, dpi=1200)
    plt.clf()


def draw_graph_xy(X, Ri, Ro, y, out,
                  cut=0.5, filename='plots/xy.png',
                  highlight_errors=False):

    # outgoing and incoming segments
    # to features, dim = (N_hits, 3)
    feats_o = torch.matmul(Ro.t(), X)
    feats_i = torch.matmul(Ri.t(), X)

    X = X.detach().numpy()
    Ri = Ri.detach().numpy()
    Ro = Ro.detach().numpy()
    y = y.detach().numpy()
    out = out.detach().numpy()
    N = len(y)
    result = np.array([((y[i] == 1 and out[i] >= cut), # true positive
                        (y[i] == 1 and out[i] <  cut), # false negative
                        (y[i] == 0 and out[i] >  cut), # false positive
                        (y[i] == 0 and out[i] <= cut)) # true negative
                       for i in range(N)])

    for i in range(len(X)):
        plt.scatter(X[i][0]*np.cos(X[i][1]*np.pi), 
                    X[i][0]*np.sin(X[i][1]*np.pi), 
                    c='silver', linewidths=0, marker='s', s=8)

    for i in range(N):
        if (result[i][0]):
            if (highlight_errors):
                plt.plot((feats_o[i][0]*np.cos(feats_o[i][1]*np.pi), 
                          feats_i[i][0]*np.cos(feats_i[i][1]*np.pi)),
                         (feats_o[i][0]*np.sin(feats_o[i][1]*np.pi), 
                          feats_i[i][0]*np.sin(feats_i[i][1]*np.pi)), 
                         'go-', lw=0.1, ms=0.1, alpha=0.5)
            else:
                plt.plot((feats_o[i][0]*np.cos(feats_o[i][1]*np.pi),
                          feats_i[i][0]*np.cos(feats_i[i][1]*np.pi)),
                         (feats_o[i][0]*np.sin(feats_o[i][1]*np.pi),
                          feats_i[i][0]*np.sin(feats_i[i][1]*np.pi)),
                         'go-', lw=0.4, ms=0.2, alpha=0.8)
            
        if (result[i][2]): # false positive
            plt.plot((feats_o[i][0]*np.cos(feats_o[i][1]*np.pi),
                      feats_i[i][0]*np.cos(feats_i[i][1]*np.pi)),
                     (feats_o[i][0]*np.sin(feats_o[i][1]*np.pi),
                      feats_i[i][0]*np.sin(feats_i[i][1]*np.pi)), 'ro-', lw=0.4, ms=0.2, alpha=0.8)
        if (result[i][1]): # false negative
            plt.plot((feats_o[i][0]*np.cos(feats_o[i][1]*np.pi),
                      feats_i[i][0]*np.cos(feats_i[i][1]*np.pi)),
                     (feats_o[i][0]*np.sin(feats_o[i][1]*np.pi),
                      feats_i[i][0]*np.sin(feats_i[i][1]*np.pi)), 'bo-', lw=0.4, ms=0, alpha=0.8)

    red_patch = mpatches.Patch(color='red', label='False Positive')
    blue_patch = mpatches.Patch(color='blue', label='False Negative')
    green_patch = mpatches.Patch(color='green', label='True Positive')
    plt.legend(handles=[green_patch, blue_patch, red_patch])
    plt.ylabel("y [m]")
    plt.xlabel("x [m]")
    plt.savefig(filename, dpi=1200)
    plt.clf()

def make_block_diagram(filename, prediction):    
    """InteractionNetwork(
        (relational_model): RelationalModel(
            (layers): Sequential(
                (0): Linear(in_features=7, out_features=150, bias=True)
                (1): ReLU()
                (2): Linear(in_features=150, out_features=150, bias=True)
                (3): ReLU()
                (4): Linear(in_features=150, out_features=150, bias=True)
                (5): ReLU()
                (6): Linear(in_features=150, out_features=1, bias=True)
                (7): Sigmoid()
            )
        )
        (object_model): ObjectModel(
            (layers): Sequential(
                (0): Linear(in_features=4, out_features=100, bias=True)
                (1): ReLU()
                (2): Linear(in_features=100, out_features=100, bias=True)
                (3): ReLU()
                (4): Linear(in_features=100, out_features=3, bias=True)
            )
        )
    )"""
    dot = make_dot(prediction) #params=dict(model.named_parameters()))
    print(dot)
    dot.format = 'png'
    dot.render(filename)


def plotDiscriminant(true_seg, false_seg, nBins, filename):
    plt.hist(true_seg, nBins, color='blue', label='Real Segments', alpha=0.7)
    plt.hist(false_seg, nBins, color='red', label='Fake Segments', alpha=0.7)
    plt.xlim(0, 1)
    plt.xlabel("Edge Weight")
    plt.ylabel("Counts")
    plt.title("Edge Weights: All Segments")
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=1200)
    plt.clf()

def confusionMatrix(true_seg, false_seg, cut):
    true_seg, false_seg = np.array(true_seg), np.array(false_seg)
    Nt, Nf = len(true_seg), len(false_seg)
    TP = len(true_seg[true_seg > cut])/Nt    # true positive
    FN = len(true_seg[true_seg < cut])/Nt    # false negative
    FP = len(false_seg[false_seg > cut])/Nf  # false positive
    TN = len(false_seg[false_seg < cut])/Nf  # true negative
    return np.array([[TP, FN], [FP, TN]])

def confusionPlot(true_seg, false_seg, filename):
    cuts = np.arange(0, 1, 0.01)
    matrices = [confusionMatrix(true_seg, false_seg, i) for i in cuts]

    plt.scatter(cuts, [matrices[i][0][0] for i in range(len(cuts))], 
                label="True Positive", color='mediumseagreen', marker='h', s=1.8)
    plt.scatter(cuts, [matrices[i][0][1] for i in range(len(cuts))],
                label="False Negative", color='orange', marker='h', s=1.8)
    plt.scatter(cuts, [matrices[i][1][0] for i in range(len(cuts))], 
                label="False Positive", color='red', marker='h', s=1.8)
    plt.scatter(cuts, [matrices[i][1][1] for i in range(len(cuts))],
                label="True Negative", color='mediumslateblue', marker='h', s=1.8)
    plt.xlabel('Discriminant Cut')
    plt.ylabel('Yield')
    plt.legend(loc='best')
    plt.savefig(filename, dpi=1200)
    plt.clf()

def plotROC(true_seg, false_seg, filename):
    cuts = np.logspace(-6, 0, 1000)
    matrices = [confusionMatrix(true_seg, false_seg, i) for i in cuts]
    plt.scatter([matrices[i][1][0] for i in range(len(cuts))], 
                [matrices[i][0][0] for i in range(len(cuts))], 
                color='goldenrod', marker='h', s=2.2)
    plt.grid(True)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(filename, dpi=1200)
    plt.clf()
