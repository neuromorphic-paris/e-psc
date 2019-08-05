import sys
import prosper
import numpy as np
from scipy.stats import truncnorm

from utils.Cards_loader import Cards_loader
from utils.Time_Surface_generators import Time_Surface_all, Time_Surface_event

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.bsc_et import BSC_ET
from prosper.em.camodels.dbsc_et import DBSC_ET
from prosper.utils import create_output_path
from prosper.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
import tables as tb
from mpi4py import MPI

import os

comm = MPI.COMM_WORLD

np.set_printoptions(threshold=sys.maxsize)
output_path = create_output_path()

#### PARAMETERS ####

learning = True  # Decide whether to run the sparse coding algorithm
classification = True  # Run classification

ts_size = 13  # size of the time surfaces
# size of the pip cards (square so dimension D = rec_size * rec_size)
rec_size = 35
tau = 5000  # time constant for the construction of time surfaces
# number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)
polarities = 1

#### IMPORTING DATASET ####
learning_set_length = 12
testing_set_length = 5

dtr = None
dte = None
if comm.rank == 0:
    fh = tb.open_file("../datasets/pokerDVS.h5")
    dtr = [d for d in fh.root.train]
    dte = [d for d in fh.root.test]
    nprocs = comm.size
    to_scatter_train = [dtr[i::nprocs] for i in range(nprocs)]
    to_scatter_test = [dte[i::nprocs] for i in range(nprocs)]

    sizes_of_train_samples = [len(dtr[j][0])
                              for j in range(len(dtr))]
    sizes_of_test_samples = [len(dtr[j][0])
                             for j in range(len(dtr))]
dtr = comm.scatter(to_scatter_train)
dte = comm.scatter(to_scatter_test)

number_of_samples = sum(sizes_of_train_samples)

number_of_features = ts_size**2
ts = np.zeros((number_of_samples, ts_size, ts_size))

idx = 0
train_labels = []
train_rec_sizes = []
for recording in range(len(dtr)):
    for k in range(dtr[recording].shape[0]):
        single_event = [dtr[recording][k, 0], [dtr[recording][k, 1],
                                               dtr[recording][k, 2]]]
        time_surface = Time_Surface_event(xdim=ts_size,
                                          ydim=ts_size,
                                          event=single_event,
                                          timecoeff=tau,
                                          dataset=dtr[recording],
                                          num_polarities=polarities,
                                          verbose=False)
        ts[idx] = time_surface
        train_labels.append(dtr[recording][k, -1])
        idx += 1
    train_rec_sizes.append(dtr[recording].shape[0])
ts = ts.reshape((ts.shape[0], -1))

number_of_samples = sum(sizes_of_test_samples)
ts_test = np.zeros((number_of_samples, ts_size, ts_size))

idx = 0
test_labels = []
test_rec_sizes = []
for recording in range(len(dte)):
    for k in range(dte[recording].shape[0]):
        single_event = [dte[recording][k, 0], [dte[recording][k, 1],
                                               dte[recording][k, 2]]]
        time_surface = Time_Surface_event(xdim=ts_size,
                                          ydim=ts_size,
                                          event=single_event,
                                          timecoeff=tau,
                                          dataset=dte[recording],
                                          num_polarities=polarities,
                                          verbose=False)
        ts_test[idx] = time_surface
        test_labels.append(dte[recording][k, -1])
        idx += 1
    test_rec_sizes.append(dte[recording].shape[0])
ts_test = ts_test.reshape((ts_test.shape[0], -1))

#### RUNNING THE SPARSE CODING ALGORITHM ####
if learning:
    # Dimensionality of the model
    H = 10     # let's start with 100
    D = ts_size**2    # dimensionality of observed data

    # Approximation parameters for Expectation Truncation (It has to be Hprime>=gamma)
    Hprime = 8
    gamma = 6

    # Import and instantiate a model
    discriminative = True
    if discriminative:
        model = DBSC_ET(D, H, Hprime, gamma)
    else:
        model = BSC_ET(D, H, Hprime, gamma)

    # Configure DataLogger
    print_list = ('T', 'L', 'pi', 'sigma')
    dlog.set_handler(print_list, TextPrinter)  # prints things to terminal
    txt_list = ('T', 'L', 'pi', 'sigma')
    dlog.set_handler(txt_list, StoreToTxt, output_path +
                     '/results.txt')  # stores things in a txt file
    h5_list = ('T', 'L', 'pi', 'sigma', 'W')
    dlog.set_handler(h5_list, StoreToH5, output_path +
                     '/results.h5')  # stores things in an h5 file

    # Choose annealing schedule
    from prosper.em.annealing import LinearAnnealing
    anneal = LinearAnnealing(120)  # decrease
    anneal['T'] = [(0, 5.), (.8, 1.)]
    anneal['Ncut_factor'] = [(0, 0.), (0.5, 0.), (0.6, 1.)]
    # anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
    # anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
    anneal['W_noise'] = [(0, np.std(ts) / 2.), (0.7, 0.)]
    # anneal['pi_noise'] = [(0,0.),(0.2,0.1),(0.7,0.)]
    anneal['anneal_prior'] = False

    train_labels = np.array(train_labels)
    assert train_labels.shape[0] == ts.shape[0]
    my_data = {'y': ts, 'l': train_labels}
    model_params = model.standard_init(my_data)
    print("model defined")
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()
    print("em finished")

    my_test_data = {'y': ts_test}
    res = model.inference(anneal, em.lparams, my_test_data)
    sparse_codes = res['s'][:, 0, :]  # should be Number of samples x H
    dlog.close()

if classification:

    my_train_data = {'y': ts}
    res_train = model.inference(anneal, em.lparams, my_train_data)

    train_features = []
    start = 0
    for i in range(len(train_rec_sizes)):
        stop = start + train_rec_sizes[i]
        train_features.append(res_train['s'][start:stop, 0, :].mean(0))
        start = stop

    train_features = np.array(train_features)

    my_test_data = {'y': ts_test}
    res_test = model.inference(anneal, em.lparams, my_test_data)

    test_features = []
    start = 0
    for i in range(len(test_rec_sizes)):
        stop = start + test_rec_sizes[i]
        test_features.append(res_test['s'][start:stop, 0, :].mean(0))
        start = stop
    test_features = np.array(test_features)

    train_features_labels = comm.gather((train_features, train_labels))
    test_features_labels = comm.gather((test_features, test_labels))
    if comm.rank == 0:
        train_features = np.concatenate([f[0] for f in train_features_labels])
        train_labels = np.concatenate([f[1] for f in train_features_labels])
        test_features = np.concatenate([f[0] for f in test_features_labels])
        test_labels = np.concatenate([f[1] for f in test_features_labels])

        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics

        lreg = LogisticRegression(solver='lbfgs', multi_class='auto')
        lreg.fit(train_features, train_labels)
        predicted_labels = lreg.predict(test_features)

        test_labels = np.array(test_labels)
        print("Classification report for classifier %s:\n%s\n"
              % (lreg, metrics.classification_report(test_labels, predicted_labels)))
        print("Confusion matrix:\n%s" %
              metrics.confusion_matrix(test_labels, predicted_labels))
