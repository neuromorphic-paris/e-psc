# events.py

# Created by Omar Oubari.
# Email: omar.oubari@inserm.fr
# Last Version: 04/09/2019

# information:
#     1. building the time surfaces (learning and testing set)
#     2. running the learning set through vc-GMM to find the cluster centers
#     3. building average histogram of activated clusters on the learning set to find the signature of each class
#     4. classification by building histogram of the test set and comparing to the learned histograms via distance (Euclidean and Bhattacharyya)

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#### PARAMETERS ####

create_features = False; # choose whether to import the dataset and create time surfaces or load from an existing npy file
save = False # choose to save the features as a .npy file
shuffle_seed = 12 # seed used for dataset shuffling, if set to 0 the process will be totally random

C=100
create_histograms = True

ts_size = 11  # size of the time surfaces
tau = 5000  # time constant for the construction of time surfaces
polarities = 1  # number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)

vc_gmm_clustering = True # choose whether to run the C++ code that clusters the dataset


def ts_info(ts):
    nts = ts+ 1e-10
    nts /= nts.sum()
    return -(np.log2(nts)*nts).sum()

if create_features:
    from utilities.Cards_loader import Cards_loader
    from utilities.Time_Surface_generators import Time_Surface_exp

    #### IMPORTING DATASET ####
    learning_set_length = 12
    testing_set_length = 5

    data_folder = "../datasets/pips/"
    dataset_learning, labels_learning, filenames_learning, dataset_testing, labels_testing, filenames_testing = Cards_loader(
        data_folder, learning_set_length, testing_set_length, shuffle_seed)

    #### BUILDING THE TRAINING DATASET ####
    sizes_of_training_samples = [len(dataset_learning[j][0]) for j in range(len(dataset_learning))]
    number_of_samples = sum(sizes_of_training_samples)

    number_of_features = ts_size**2

    ts_train = []
    train_labels = []
    pip_trdpt = []
    for recording in range(len(dataset_learning)):
        idx = 0
        for k in range(len(dataset_learning[recording][0])):
            single_event = [dataset_learning[recording]
                            [0][k], dataset_learning[recording][1][k]]

            # exponential time surfaces
            time_surface = Time_Surface_exp(xdim=ts_size,
                                            ydim=ts_size,
                                            event=single_event,
                                            timecoeff=tau,
                                            dataset=dataset_learning[recording],
                                            num_polarities=polarities,
                                            minv=0.1,
                                            verbose=False)
                                            
            if ts_info(time_surface)>1: # improvement needed for minimum activity
                ts_train.append(time_surface)
                train_labels.append(labels_learning[recording])
                idx += 1
        pip_trdpt.append(idx)
        
    pip_trdpt = np.array(pip_trdpt, np.int32)
    ts_train = np.array(ts_train, np.float32)
    ts_train = ts_train.reshape((ts_train.shape[0], -1))
    train_labels = np.array(train_labels, np.int32)

    #### BUILDING THE TESTING DATASET ####
    sizes_of_testing_samples = [len(dataset_testing[j][0]) for j in range(len(dataset_testing))]

    number_of_samples = sum(sizes_of_testing_samples)
    ts_test = []
    test_labels = []
    pip_tedpt = []
    for recording in range(len(dataset_testing)):
        idx = 0
        for k in range(len(dataset_testing[recording][0])):
            single_event = [dataset_testing[recording]
                            [0][k], dataset_testing[recording][1][k]]

            # exponential time surfaces
            time_surface = Time_Surface_exp(xdim=ts_size,
                                            ydim=ts_size,
                                            event=single_event,
                                            timecoeff=tau,
                                            dataset=dataset_testing[recording],
                                            num_polarities=polarities,
                                            minv=0.1,
                                            verbose=False)

            if ts_info(time_surface)>1: # improvement needed for minimum activity
                ts_test.append(time_surface)
                test_labels.append(labels_testing[recording])
                idx += 1
        pip_tedpt.append(idx)
        
    pip_tedpt = np.array(pip_tedpt, np.int32)
    ts_test = np.array(ts_test, np.float32)
    ts_test = ts_test.reshape((ts_test.shape[0], -1))
    test_labels = np.array(test_labels, np.int32)

    if save:
        np.save('features/poker_ts_train.npy', ts_train) # save the training features as an npy file
        np.save('features/poker_ts_test.npy', ts_test) # save the test features as an npy file

        np.save('features/poker_train_labels.npy', train_labels) # save the training labels as a text file
        np.save('features/poker_test_labels.npy', test_labels) # save the test labels as an npy file

        np.save('features/poker_train_nts.npy', pip_trdpt) # save the training labels as an npy file
        np.save('features/poker_test_nts.npy', pip_tedpt) # save the test labels as an npy file

#### VC-GMM CLUSTERING ON LEARNING DATASET ####
if vc_gmm_clustering:
    import subprocess

    cmd_list = ["build/release/events",        # path to the C++ executable for clustering
                "features/poker_ts_train.npy", # path to the training features (saved in a text file)
                "features/poker_ts_test.npy",  # path to the test features (saved in a text file)
                "4",                           # int - C_p - number of clusters considered for each data point
                "4",                           # int - G - search space (nearest neighbours for the C' clusters)
                "1",                           # bool - plus1 - include one additional randomly chosen cluster
                "2000",                        # int - N_core - size of subset
                str(C),                        # int - C - number of cluster centers
                "5",                           # int - chain_length - chain length for AFK-MC² seeding
                "0.0001",                      # float - convergence_threshold
                "1",                           # bool - save_centers - write cluster centers to a text file
                "1",                           # bool - save_prediction - write assigned clusters to a text file
               ]
    subprocess.call(cmd_list)

#### AVERAGE HISTOGRAM OF ACTIVATED CLUSTERS ####

if create_histograms:
    from utilities.Cards_loader import Cards_loader
    #### IMPORTING DATASET ####
    learning_set_length = 12
    testing_set_length = 5

    data_folder = "../datasets/pips/"
    dataset_learning, labels_learning, filenames_learning, dataset_testing, labels_testing, filenames_testing = Cards_loader(
        data_folder, learning_set_length, testing_set_length, shuffle_seed)

    #### BUILDING THE TRAINING DATASET ####
    sizes_of_training_samples = [len(dataset_learning[j][0]) for j in range(len(dataset_learning))]
    number_of_samples = sum(sizes_of_training_samples)

    number_of_features = ts_size**2
    ts_train = np.zeros((number_of_samples, ts_size, ts_size))

    tef= np.load("gmm_test_labels.npy")
    trf= np.load("gmm_train_labels.npy")
    trl= np.load("features/poker_train_labels.npy")
    tel= np.load("features/poker_test_labels.npy")
    trdpt= np.load("features/poker_train_nts.npy")
    tedpt= np.load("features/poker_test_nts.npy")
    
    assert trdpt.sum()==trf.shape[0]
    assert trdpt.sum()==trl.shape[0]
    assert tedpt.sum()==tef.shape[0]
    assert tedpt.sum()==tel.shape[0]

    ### Training Features ####
    train_labels = []
    start = 0 
    trfeats = []
    trlabel = []
    for r in range(trdpt.shape[0]):
        stop = start + trdpt[r]
        this_f=np.zeros((stop-start,C))
        for k in range(trdpt[r]):
            this_f[k,trf[start+k]]=1
        trfeats.append(this_f.mean(0))
        assert (trl[start:stop]==trl[start]).all()
        trlabel.append(trl[start])
        start = stop
    trfeats = np.array(trfeats)
    trlabel = np.array(trlabel)

    #### BUILDING THE TESTING DATASET ####
    sizes_of_testing_samples = [len(dataset_testing[j][0]) for j in range(len(dataset_testing))]

    number_of_samples = sum(sizes_of_testing_samples)
    ts_test = np.zeros((number_of_samples, ts_size, ts_size))

    ### Testing Features ####
    test_labels = []
    start = 0 
    stop = 0
    tefeats = []
    telabel = []
    for r in range(tedpt.shape[0]):
        stop = start+tedpt[r]
        this_f=np.zeros((stop-start,C))
        for k in range(tedpt[r]):
            this_f[k,tef[start+k]]=1
        tefeats.append(this_f.mean(0))
        assert (tel[start:stop]==tel[start]).all()
        telabel.append(tel[start])
        start = stop

    tefeats = np.array(tefeats)
    telabel = np.array(telabel)


    from sklearn.naive_bayes import GaussianNB
    from sklearn import metrics

    gnb = GaussianNB()
    gnb.fit(trfeats,trlabel)

    gnb_pl = gnb.predict(tefeats)
    gnb_pl_tr = gnb.predict(trfeats)

    print("GaussianNB Classification report for classifier %s:\n%s\n"
          % (gnb, metrics.classification_report(telabel, gnb_pl)))
    print("GaussianNB Confusion matrix:\n%s" %
          metrics.confusion_matrix(telabel, gnb_pl))

    print("GaussianNB Classification report for classifier on training %s:\n%s\n"
          % (gnb, metrics.classification_report(trlabel, gnb_pl_tr)))
    print("GaussianNB Confusion matri on trainingx:\n%s" %
          metrics.confusion_matrix(trlabel, gnb_pl_tr))

    from sklearn.linear_model import LogisticRegression

    lreg = LogisticRegression()
    lreg.fit(trfeats,trlabel)

    lreg_pl = lreg.predict(tefeats)
    lreg_pl_tr = lreg.predict(trfeats)

    print("LogisticRegression Classification report for classifier %s:\n%s\n"
          % (lreg, metrics.classification_report(telabel, lreg_pl)))
    print("LogisticRegression Confusion matrix:\n%s" %
          metrics.confusion_matrix(telabel, lreg_pl))

    print("LogisticRegression Classification report for classifier on training %s:\n%s\n"
          % (lreg, metrics.classification_report(trlabel, lreg_pl_tr)))
    print("LogisticRegression Confusion matrix on training:\n%s" %
          metrics.confusion_matrix(trlabel, lreg_pl_tr))


    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(trfeats,trlabel)

    knn_pl = knn.predict(tefeats)
    knn_pl_tr = knn.predict(trfeats)

    print("KNeighborsClassifier Classification report for classifier %s:\n%s\n"
          % (knn, metrics.classification_report(telabel, knn_pl)))
    print("KNeighborsClassifier Confusion matrix:\n%s" %
          metrics.confusion_matrix(telabel, knn_pl))

    print("KNeighborsClassifier Classification report for classifier on training %s:\n%s\n"
          % (knn, metrics.classification_report(trlabel, knn_pl_tr)))
    print("KNeighborsClassifier Confusion matrix on training:\n%s" %
          metrics.confusion_matrix(trlabel, knn_pl_tr))

    from sklearn.svm import SVC

    svc = SVC(kernel='linear', C=0.025)
    svc.fit(trfeats,trlabel)

    svc_pl = svc.predict(tefeats)
    svc_pl_tr = svc.predict(trfeats)

    print("Support Vector Classification report for classifier %s:\n%s\n"
          % (svc, metrics.classification_report(telabel, svc_pl)))
    print("Support Vector Confusion matrix:\n%s" %
          metrics.confusion_matrix(telabel, svc_pl))

    print("Support Vector Classification report for classifier on training %s:\n%s\n"
          % (svc, metrics.classification_report(trlabel, svc_pl_tr)))
    print("Support Vector Confusion matrix on training:\n%s" %
          metrics.confusion_matrix(trlabel, svc_pl_tr))
