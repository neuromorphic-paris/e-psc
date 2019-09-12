import numpy as np
from Cards_loader import Cards_loader
from Time_Surface_generators import Time_Surface_exp, Time_Surface_gauss

#### PARAMETERS ####
clustering = False # choose whether to run the C++ code that clusteers the dataset
gaussian_ts = False # choose between exponential time surfaces and gaussian time surfaces
ts_size = 13  # size of the time surfaces
tau = 5000  # time constant for the construction of time surfaces
polarities = 1  # number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)

#### IMPORTING DATASET ####
learning_set_length = 12
testing_set_length = 50

data_folder = "../../datasets/pips/"
dataset_learning, labels_learning, filenames_learning, dataset_testing, labels_testing, filenames_testing = Cards_loader(
    data_folder, learning_set_length, testing_set_length)

#### BUILDING THE TRAINING DATASET ####
sizes_of_training_samples = [len(dataset_learning[j][0]) for j in range(len(dataset_learning))]
number_of_samples = sum(sizes_of_training_samples)

number_of_features = ts_size**2
ts = np.zeros((number_of_samples, ts_size, ts_size))

idx = 0
training_labels = []
for recording in range(len(dataset_learning)):
    for k in range(len(dataset_learning[recording][0])):
        single_event = [dataset_learning[recording]
                        [0][k], dataset_learning[recording][1][k]]

        if gaussian_ts:
          # gaussian time surfaces
          time_surface = Time_Surface_gauss(xdim=ts_size,
                                            ydim=ts_size,
                                            event=single_event,
                                            sigma=100,
                                            dataset=dataset_learning[recording],
                                            num_polarities=polarities,
                                            minv=0.1,
                                            verbose=False)
        else:
          # exponential time surfaces
          time_surface = Time_Surface_exp(xdim=ts_size,
                                            ydim=ts_size,
                                            event=single_event,
                                            timecoeff=tau,
                                            dataset=dataset_learning[recording],
                                            num_polarities=polarities,
                                            minv=0.1,
                                            verbose=False)

        ts[idx] = time_surface
        training_labels.append(recording)
        idx += 1
ts = ts.reshape((ts.shape[0], -1))

#### BUILDING THE TESTING DATASET ####
sizes_of_testing_samples = [len(dataset_testing[j][0]) for j in range(len(dataset_testing))]

number_of_samples = sum(sizes_of_testing_samples)
ts_test = np.zeros((number_of_samples, ts_size, ts_size))

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
        ts_test[idx] = time_surface
        test_labels.append(recording)
        idx += 1
ts_test = ts_test.reshape((ts_test.shape[0], -1))

#### RUNNING THE CLUSTERING ALGORITHM ON THE TRAINING DATASET
if clustering:

    import subprocess

    cmd_list = ["../build/release/example", # path to the C++ executable for clustering
               ]
    subprocess.call(cmd_list)
