#### PARAMETERS ####
create_features = False; # choose whether to import the dataset and create time surfaces or load from an existing npy file
gaussian_ts = False # choose between exponential time surfaces and gaussian time surfaces
ts_size = 13  # size of the time surfaces
tau = 5000  # time constant for the construction of time surfaces
polarities = 1  # number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)

clustering = True # choose whether to run the C++ code that clusteers the dataset

if create_features:
    import numpy as np
    from utilities.Cards_loader import Cards_loader
    from utilities.Time_Surface_generators import Time_Surface_exp, Time_Surface_gauss

    #### IMPORTING DATASET ####
    learning_set_length = 12
    testing_set_length = 5

    data_folder = "../datasets/pips/"
    dataset_learning, labels_learning, filenames_learning, dataset_testing, labels_testing, filenames_testing = Cards_loader(
        data_folder, learning_set_length, testing_set_length)

    #### BUILDING THE TRAINING DATASET ####
    sizes_of_training_samples = [len(dataset_learning[j][0]) for j in range(len(dataset_learning))]
    number_of_samples = sum(sizes_of_training_samples)

    number_of_features = ts_size**2
    ts_train = np.zeros((number_of_samples, ts_size, ts_size))

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

            if np.count_nonzero(time_surface):
                ts_train[idx] = time_surface
                training_labels.append(recording)
                idx += 1
    ts_train = ts_train.reshape((ts_train.shape[0], -1))

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

            if np.count_nonzero(time_surface):
                ts_test[idx] = time_surface
                test_labels.append(recording)
                idx += 1
    ts_test = ts_test.reshape((ts_test.shape[0], -1))

    np.savetxt('poker_ts_train.txt', ts_train) # save the training features
    np.savetxt('poker_ts_test.txt', ts_test) # save the test features

#### RUNNING THE CLUSTERING ALGORITHM ON THE TRAINING DATASET
if clustering:
    import subprocess

    cmd_list = ["build/release/events_gmm",    # path to the C++ executable for clustering
                "poker_ts_train.txt",          # path to the training features (saved in a text file)
                "poker_ts_test.txt",           # path to the test features (saved in a text file)
                "4",                           # int - C_p - number of clusters considered for each data point
                "4",                           # int - G - search space (nearest neighbours for the C' clusters)
                "1",                           # bool - plus1 - include one additional randomly chosen cluster to each of the search spaces
                "10000",                       # int - N_core - size of subset
                "4",                           # int - C - number of cluster centers
                "20",                          # int - chain_length - chain length for AFK-MC² seeding
                "0.0001",                      # float - convergence_threshold
                "1",                           # bool - save - write cluster centers to a text file
               ]
    subprocess.call(cmd_list)
