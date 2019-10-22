import sys
import prosper
import numpy as np
from scipy.stats import truncnorm

import sys
sys.path.append("..")
from utils.Cards_loader import Cards_loader
from utils.Time_Surface_generators import Time_Surface_all, Time_Surface_exp

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.dsc_et import DSC_ET
#from prosper.em.camodels.bsc_et import BSC_ET
#from prosper.em.camodels.dbsc_et import DBSC_ET
from prosper.utils import create_output_path
from prosper.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
import tables as tb
from mpi4py import MPI

import os, sys

comm = MPI.COMM_WORLD

nprocs = comm.size

def pp(msg,rank=-1,comm=comm):
    if rank==-1:
        print(msg)
        sys.stdout.flush()
    elif comm.rank==rank:
        print(msg)
        sys.stdout.flush()

#pp("running {} parallel processes".format(nprocs))
pp(sys.version_info,0)

def shinfo(x):
    nx = x.copy()+1e-10
    nx/=nx.sum()
    return (-nx*np.log2(nx)).sum()
# PARAMETERS ####

learning = True# Decide whether to run the sparse coding algorithm
classification = True  # Run classification
resume  = False #True
ts_size = 11  # size of the time surfaces
tau = 5000  # time constant for the construction of time surfaces
polarities = 1  # number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)

if resume:
    output_path = "/users/ml/xoex6879/workspace/psc/prosper_stuff/output/main_nmnist.py.d1062928/"
else:
    output_path = create_output_path()

dtr = None
dte = None
#pp("rank: ", comm.rank)

to_scatter_train = None
to_scatter_test = None
if comm.rank == 0:
    #fh = tb.open_file("../datasets/nmnist_small.h5")
    fh = tb.open_file("../../../datasets/nmnist_one_saccade.h5")
    #fh = tb.open_file("../datasets/nmnist_one_saccade.h5")
    #fh = tb.open_file("../datasets/nmnist.h5")
    dtr = [d.read().astype(np.int32) for d in fh.root.train]
    dte = [d.read().astype(np.int32) for d in fh.root.test]

    to_scatter_train = [dtr[i::nprocs] for i in range(nprocs)]
    to_scatter_test = [dte[i::nprocs] for i in range(nprocs)]
    fh.close()
#import ipdb;ipdb.set_trace()
dtr = comm.scatter(to_scatter_train)
dte = comm.scatter(to_scatter_test)
pp("1st rank: {}, len(dtr): {}, len(dte): {}".format(comm.rank, len(dtr), len(dte)))
sys.stdout.flush()
# number_of_samples = sum(sizes_of_train_samples)


# idx = 0
ts = []
train_labels = []
train_rec_sizes = []
for recording in range(len(dtr)):
    idx=0
    for k in range(dtr[recording].shape[0]):
        single_event = [dtr[recording][k, 0].astype(np.int),
                        dtr[recording][k, 1:3].astype(np.int)]
        dataset = [dtr[recording][:, 0].astype(np.int),
                   dtr[recording][:, 1:3].astype(np.int),
                   dtr[recording][:, 3].astype(np.int)-1]

        time_surface = Time_Surface_exp(xdim=ts_size,
                                          ydim=ts_size,
                                          event=single_event,
                                          timecoeff=tau,
                                          dataset=dataset,
                                          num_polarities=polarities,
                                          verbose=False)
        if shinfo(time_surface)>3:
            ts.append(time_surface)
            train_labels.append(int(dtr[recording][k, -1]))
        idx += 1
    train_rec_sizes.append(idx)
ts = np.array(ts)
ts = ts.reshape((ts.shape[0], -1))
train_labels = np.array(train_labels)
# ts_res = ts.shape[0] % comm.size
# ts = ts[:-ts_res]
# train_labels = train_labels[:-ts_res]
# pp(len(train_labels))

ts_test = []
test_labels = []
test_rec_sizes = []
for recording in range(len(dte)):
    idx= 0 
    for k in range(dte[recording].shape[0]):

        single_event = [dte[recording][k, 0].astype(np.int),
                        dte[recording][k, 1:3].astype(np.int)]
        dataset = [dte[recording][:, 0].astype(np.int),
                   dte[recording][:, 1:3].astype(np.int),
                   dte[recording][:, 3].astype(np.int)-1]
        time_surface = Time_Surface_exp(xdim=ts_size,
                                          ydim=ts_size,
                                          event=single_event,
                                          timecoeff=tau,
                                          dataset=dataset,
                                          num_polarities=polarities,
                                          verbose=False)
        if shinfo(time_surface)>3:
            ts_test.append(time_surface)
            test_labels.append(int(dte[recording][k, -1]))
        idx += 1
    test_rec_sizes.append(dte[recording].shape[0])
ts_test = np.array(ts_test)
ts_test = ts_test.reshape((ts_test.shape[0], -1))
test_labels = np.array(test_labels)
# ts_test_res = ts_test.shape[0] % comm.size
# ts_test = ts_test[:-ts_test_res]
# test_labels = test_labels[:-ts_test_res]
pp("2nd rank: {}, ts.shape: {}, train_labels.shape: {}".format(comm.rank, ts.shape, train_labels.shape))
sys.stdout.flush()
comm.barrier()
#### RUNNING THE SPARSE CODING ALGORITHM ####
if learning:
    # Dimensionality of the model
    H = 1000     # let's start with 100
    D = ts_size**2    # dimensionality of observed data

    # Approximation parameters for Expectation Truncation (It has to be Hprime>=gamma)
    Hprime = 3
    gamma = 3
    states = np.array([0,1])

    # Import and instantiate a model
    discriminative = False
    if discriminative:
        model = DBSC_ET(D, H, Hprime, gamma)
    else:
        model = DSC_ET(D, H, Hprime, gamma, states=states)

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
    anneal = LinearAnnealing(50)  # decrease
    anneal['T'] = [(0, 5.), (.8, 1.)]
    anneal['Ncut_factor'] = [(0, 0.), (0.5, 0.), (0.6, 1.)]
    # anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
    # anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
    anneal['W_noise'] = [(0, np.std(ts) / 2.), (0.7, 0.)]
    # anneal['pi_noise'] = [(0,0.),(0.2,0.1),(0.7,0.)]
    anneal['anneal_prior'] = False

    assert train_labels.shape[0] == ts.shape[0]
    my_data = {'y': ts, 'l': train_labels}
    model_params = model.standard_init(my_data)
    pp("model defined",0)
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run(verbose=True)
    pp("em finished",0)

    my_test_data = {'y': ts_test}
    res = model.inference(anneal, em.lparams, my_test_data)
    sparse_codes = res['s'][:, 0, :]  # should be Number of samples x H
    dlog.close()
    model_params = em.lparams

if resume:
    res_file = output_path + 'results.h5'
    with tb.open_file(res_file) as fh:
        W = fh.root.W[-1]
        pi = fh.root.pi[-1]
        sigma = fh.root.sigma[-1]

    # Dimensionality of the model
    H = 500     # let's start with 100
    D = ts_size**2    # dimensionality of observed data

    # Approximation parameters for Expectation Truncation
    # (It has to be Hprime>=gamma)
    Hprime = 3
    gamma = 3
    states = np.array([0,1])

    # Import and instantiate a model
    discriminative = False
    if discriminative:
        model = DBSC_ET(D, H, Hprime, gamma)
    else:
        model = DSC_ET(D, H, Hprime, gamma, states=states)
    model_params = {"W": W, "pi": pi, "sigma": sigma}

    # Choose annealing schedule
    from prosper.em.annealing import LinearAnnealing
    anneal = LinearAnnealing(20)  # decrease
    anneal['T'] = [(0, 5.), (.8, 1.)]
    anneal['Ncut_factor'] = [(0, 0.), (0.5, 0.), (0.6, 1.)]
    # anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
    # anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
    anneal['W_noise'] = [(0, np.std(ts) / 2.), (0.7, 0.)]
    # anneal['pi_noise'] = [(0,0.),(0.2,0.1),(0.7,0.)]
    anneal['anneal_prior'] = False
    anneal.cur_pos=20

if classification:

    my_train_data = {'y': ts}
    pp(Hprime,gamma)
    res_train = model.inference(anneal, model_params, my_train_data,
                                Hprime_max=Hprime, gamma_max=gamma)

    train_features = []
    train_labels2 = []
    start = 0
    for i in range(len(train_rec_sizes)):
        stop = start + train_rec_sizes[i]
        train_features.append(res_train['s'][start:stop, 0, :].mean(0))
        this_l = train_labels[start:stop]
        assert (this_l == this_l[0]).all()
        train_labels2.append(this_l[0])
        start = stop

    train_features = np.array(train_features)
    train_labels = np.array(train_labels2)

    my_test_data = {'y': ts_test}
    res_test = model.inference(anneal, model_params, my_test_data,
                               Hprime_max=Hprime, gamma_max=gamma)

    test_features = []
    test_labels2 = []
    start = 0
    for i in range(len(test_rec_sizes)):
        stop = start + test_rec_sizes[i]
        test_features.append(res_test['s'][start:stop, 0, :].mean(0))
        this_l = test_labels[start:stop]
        assert (this_l == this_l[0]).all()
        test_labels2.append(this_l[0])
        start = stop
    test_features = np.array(test_features)
    test_labels = np.array(test_labels2)

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
        pp("Classification report for classifier %s:\n%s\n"
              % (lreg, metrics.classification_report(test_labels, predicted_labels)))
        pp("Confusion matrix:\n%s" %
              metrics.confusion_matrix(test_labels, predicted_labels))
