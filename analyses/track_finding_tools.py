import queue 

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

import analyses.prepare_plots as plots

def DBScan(dist_matrix, eps=0.25, min_pts=1):
    NOISE, CORE, BORDER, CLUSTER = 0, -1, -2, 1
    point_labels = np.zeros(dist_matrix.shape[0])
    core_points     = []
    non_core_points = []
    neighbor_pts    = []
    neighbor_dists  = []
    
    n_hits = dist_matrix.shape[0]
    for h in range(n_hits):
        o, i = dist_matrix[h, :], dist_matrix[:, h]
        i_close = (i > 0) & (i <= eps)
        o_close = (o > 0) & (o <= eps)
        N_eps = i_close | o_close
        N_eps_dists = np.concatenate((i[i_close], o[o_close]))
        N_eps_idxs  = np.arange(n_hits)[N_eps]
        neighbor_dists.append(N_eps_dists)
        neighbor_pts.append(N_eps_idxs)
        if (N_eps_idxs.shape[0] >= min_pts):
            point_labels[h] = CORE
            core_points.append(h)
        else: non_core_points.append(h)

    for h in non_core_points:
        for p in neighbor_pts[h]:
            if p in core_points:
                point_labels[h] = BORDER
                break

    for i, label in enumerate(point_labels):
        q = queue.Queue()
        if (label == CORE):
            point_labels[i] == CLUSTER
            for j in neighbor_pts[i]:
                if (point_labels[j] == CORE):
                    q.put(j)
                    point_labels[j] = CLUSTER
                elif (point_labels[j] == BORDER):
                    point_labels[j] = CLUSTER
            while not q.empty():
                neighbors = neighbor_pts[q.get()]
                for k in neighbors:
                    if (point_labels[k] == CORE):
                        point_labels[k] = CLUSTER
                        q.put(k)
                    if (point_labels[k] == BORDER):
                        point_labels[k] = CLUSTER
            CLUSTER += 1

    return point_labels


def plot_N_clusters(N, X, feats_i, feats_o, hit_clusters, filename):
    
    X = np.array(X)
    X[:, 1] *= np.sign(X[:, 2])
    colors = cm.gist_ncar(np.linspace(0, 1, N))
    for i in range(len(X)):
        if (hit_clusters[i] < N):
            plt.scatter(X[i][3], X[i][1], c=colors[int(hit_clusters[i])],
                        linewidths=0, marker='s', s=8)

    for i in range(len(feats_o)):
        plt.plot((feats_o[i][2], feats_i[i][2]),
                 (feats_o[i][0], feats_i[i][0]),
                 'bo-', lw=0.1, ms=0.1, alpha=0.3)

    plt.ylabel("R [m]")
    plt.xlabel("z [m]")
    plt.savefig(filename, dpi=1200)
    plt.show()
    plt.clf()


def get_k_dists(dist_matrix):
    one_dists, two_dists, three_dists = [], [], []
    for h in range(dist_matrix.shape[0]):
        o = dist_matrix[h, :]
        i = dist_matrix[:, h]
        non_zeros = np.concatenate((i[np.nonzero(i)], o[np.nonzero(o)]))
        if len(non_zeros) > 0:
            non_zeros = np.sort(non_zeros)
            one_dists.append(non_zeros[0])
            if len(non_zeros) > 1:
                two_dists.append(non_zeros[1])
                if len(non_zeros) > 2:
                    three_dists.append(non_zeros[2])
                    
    return one_dists, two_dists, three_dists
                    
def plot_k_dists(one_dists, two_dists, three_dists):
    plots.plotSingleHist(one_dists, "1-dists", "counts",
                         16, color='mediumseagreen')
    plots.plotSingleHist(two_dists, "2-dists", "counts",
                         16, color='mediumseagreen')
    plots.plotSingleHist(three_dists, "3-dists", "counts",
                         16, color='mediumseagreen')

    one_dists = np.sort(one_dists)
    plt.plot(np.flip(one_dists))
    plt.ylabel("1-distance")
    plt.show()
    plt.clf()
    two_dists = np.sort(two_dists)
    plt.plot(np.flip(two_dists))
    plt.ylabel("2-distance")
    plt.show()
    plt.clf()
    three_dists = np.sort(three_dists)
    plt.plot(np.flip(three_dists))
    plt.ylabel("3-distance")
    plt.show()
    plt.clf()



