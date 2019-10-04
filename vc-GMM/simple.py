# simple.py
# IMPORTANT:ONLY WORKS WITH THE GMM_branch branch on the github (the inference function is different)

# Created by Omar Oubari.
# Email: omar.oubari@inserm.fr
# Last Version: 04/09/2019

# information: testing the vc-GMM clustering

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

cmd_list = ["build/release/simple",        # path to the C++ executable for clustering
            "datasets/simple/data.txt",    # path to the training features (saved in a text file)
            "15",                          # int - C_p - number of clusters considered for each data point
            "15",                          # int - G - search space (nearest neighbours for the C' clusters)
            "1",                           # bool - plus1 - include one additional randomly chosen cluster
            "1000",                        # int - N_core - size of subset
            "15",                          # int - C - number of cluster centers
            "20",                          # int - chain_length - chain length for AFK-MC² seeding
            "0.0001",                      # float - convergence_threshold
            "1",                           # bool - save_centers - write cluster centers to a text file
            "1",                           # bool - save_prediction - write assigned clusters to a text file
            ]

subprocess.call(cmd_list)

### plot grounds truth centers VS GMM centers
gt_centers = np.loadtxt("datasets/simple/gt_centers.txt")
gmm_centers = np.loadtxt("gmm_centers.txt")

fig = plt.scatter(gt_centers[:,0],gt_centers[:,1])
fig = plt.scatter(gmm_centers[:,0],gmm_centers[:,1])

# find closest gt cluster and assign the proper index
kd_tree = spatial.KDTree(gt_centers)
gmm_center_labels = [kd_tree.query(center)[1] + 1 for center in gmm_centers]

# correct inferred center indices according to match the ground truth indexing
gt_labels = np.loadtxt("datasets/simple/gt_labels.txt")
gmm_output = np.loadtxt("gmm_labels.txt")[:,1:]

gmm_corrected_labels = [gmm_center_labels[np.where((gmm_centers[:,0]==inferred_center[0]) & (gmm_centers[:,1]==inferred_center[1]))[0][0]] for inferred_center in gmm_output]

clustering_error = np.mean(gt_labels != gmm_corrected_labels)
print("clustering error",clustering_error)

plt.show()
