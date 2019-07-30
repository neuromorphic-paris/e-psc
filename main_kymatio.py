import sys
import numpy as np
from scipy.stats import truncnorm

from utils.Cards_loader import Cards_loader as Cards
from utils.Time_Surface_generators import Time_Surface_all, Time_Surface_event

from kymatio import Scattering2D
import os
import torch
from time import time
np.set_printoptions(threshold=sys.maxsize)


#### PARAMETERS ####

learning = True  # Decide whether to run the sparse coding algorithm
classification = True  # Run classification

ts_size = 17  # size of the time surfaces
# size of the pip cards (square so dimension D = rec_size * rec_size)
rec_size = 35
tau = 5000  # time constant for the construction of time surfaces
# number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)
polarities = 1

#### IMPORTING DATASET ####
learning_set_length = 12
testing_set_length = 5

data_folder = "datasets/pips/"
dataset_learning, labels_training, dataset_testing, labels_testing = Cards(
    data_folder, learning_set_length, testing_set_length)

#### BUILDING THE LEARNING DATASET ####
sizes_of_training_samples = [len(dataset_learning[j][0])
                             for j in range(len(dataset_learning))]
number_of_samples = sum(sizes_of_training_samples)

number_of_features = ts_size**2
ts = np.zeros((number_of_samples, ts_size-1, ts_size-1))

idx = 0
training_labels = []
for recording in range(len(dataset_learning)):
    for k in range(len(dataset_learning[recording][0])):
        single_event = [dataset_learning[recording]
                        [0][k], dataset_learning[recording][1][k]]
        time_surface = Time_Surface_event(xdim=ts_size,
                                          ydim=ts_size,
                                          event=single_event,
                                          timecoeff=tau,
                                          dataset=dataset_learning[recording],
                                          num_polarities=polarities,
                                          verbose=False)
        ts[idx] = time_surface[:-1,:-1]
        training_labels.append(recording)
        idx += 1
ts = ts.reshape((ts.shape[0], -1))

#### BUILDING THE TESTING DATASET ####
sizes_of_testing_samples = [len(dataset_testing[j][0])
                            for j in range(len(dataset_testing))]

number_of_samples = sum(sizes_of_testing_samples)
ts_test = np.zeros((number_of_samples, ts_size-1, ts_size-1))

idx = 0
test_labels = []
for recording in range(len(dataset_testing)):
    for k in range(len(dataset_testing[recording][0])):
        single_event = [dataset_testing[recording]
                        [0][k], dataset_testing[recording][1][k]]
        time_surface = Time_Surface_event(xdim=ts_size,
                                          ydim=ts_size,
                                          event=single_event,
                                          timecoeff=tau,
                                          dataset=dataset_testing[recording],
                                          num_polarities=polarities,
                                          verbose=False)
        ts_test[idx] = time_surface[:-1,:-1]
        test_labels.append(recording)
        idx += 1
ts_test = ts_test.reshape((ts_test.shape[0], -1))

# RUNNING THE SPARSE CODING ALGORITHM #
if learning:
    sz = int(np.sqrt(ts.shape[-1]))
    t1 = time()
    ts = torch.from_numpy(ts.reshape((ts.shape[0], 1, sz, sz)).astype(np.float32))
    ts_test = torch.from_numpy(ts_test.reshape((ts_test.shape[0], 1, sz, sz)).astype(np.float32))
    sc = Scattering2D(J=3, shape=(sz, sz))
    if torch.cuda.is_available():
        print("Using Cuda")
        sc.cuda()
        ts.cuda()
        ts_test.cuda()
    else:
        print("Not Using cuda")
    train_coef = sc(ts)
    test_coef = sc(ts_test)
    if torch.cuda.is_available():
        train_coef = train_coef.cpu()
        test_coef = test_coef.cpu()
    t2 = time()
    print("Scattering lasted {}s".format(t2 - t1))


if classification:

    train_features = []
    start = 0
    for i in range(len(sizes_of_training_samples)):
        stop = start + sizes_of_training_samples[i]
        train_features.append(train_coef[start:stop, :].mean(0))
        start = stop

    train_features = torch.cat(train_features).numpy()
    train_features = train_features.reshape((train_features.shape[0], -1))

    test_features = []
    start = 0
    for i in range(len(sizes_of_testing_samples)):
        stop = start + sizes_of_testing_samples[i]
        test_features.append(test_coef[start:stop, :].mean(0))
        start = stop

    test_features = torch.cat(test_features).numpy()
    test_features = test_features.reshape((test_features.shape[0], -1))

    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    lreg = LogisticRegression()
    lreg.fit(train_features, labels_training)
    predicted_labels = lreg.predict(test_features)

    print("Classification report for classifier %s:\n%s\n"
          % (lreg, metrics.classification_report(labels_testing,
                                                 predicted_labels)))
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(labels_testing, predicted_labels))
