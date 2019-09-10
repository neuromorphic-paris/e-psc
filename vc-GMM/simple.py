# simple.py

# Created by Omar Oubari.
# Email: omar.oubari@inserm.fr
# Last Version: 04/09/2019

# information: testing the vc-GMM clustering

import subprocess
import numpy as np
import matplotlib.pyplot as plt

#cmd_list = ["build/release/events",        # path to the C++ executable for clustering
#            "features/poker_ts_train.txt", # path to the training features (saved in a text file)
#            "features/poker_ts_test.txt",  # path to the test features (saved in a text file)
#            "15",                          # int - C_p - number of clusters considered for each data point
#            "15",                          # int - G - search space (nearest neighbours for the C' clusters)
#            "1",                           # bool - plus1 - include one additional randomly chosen cluster
#            "1000",                        # int - N_core - size of subset
#            "15",                          # int - C - number of cluster centers
#            "20",                          # int - chain_length - chain length for AFK-MC² seeding
#            "0.0001",                      # float - convergence_threshold
#            "1",                           # bool - save - write cluster centers to a text file
#            ]
#
#subprocess.call(cmd_list)

# plot grounds truth centers VS GMM centers
gt_centers = np.loadtxt("datasets/simple/gt_centers.txt")
gmm_centers = np.loadtxt("build/release/gmm_centers.txt")

fig = plt.scatter(gt_centers[:,0],gt_centers[:,1])
fig = plt.scatter(gmm_centers[:,0],gmm_centers[:,1])

plt.show()
