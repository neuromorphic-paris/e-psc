import sys
import prosper 
import numpy as np

from utils.Cards_loader import Cards_loader
from utils.Time_Surface_generators import Time_Surface_all, Time_Surface_event

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.bsc_et import BSC_ET
import os 

np.set_printoptions(threshold=sys.maxsize)

#### PARAMETERS ####

learning = True # Decide whether to run the sparse coding algorithm
ts_size = 11 # size of the time surfaces
rec_size = 35 # size of the pip cards (square so dimension D = rec_size * rec_size)
tau = 5000 # time constant for the construction of time surfaces
polarities = 1 # number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)

#### BUILDING THE LEARNING AND TESTING DATASETS ####
learning_set_length = 2
testing_set_length = 5

data_folder = "datasets/pips/"
dataset_learning, labels_learning, dataset_testing, labels_testing = Cards_loader(data_folder, learning_set_length,testing_set_length)

# number of features  
number_of_features = rec_size**2

# setting up the learning dataset
number_of_samples = len(dataset_learning)
ts = np.zeros((number_of_samples,number_of_features,ts_size,ts_size))
for recording in range(len(dataset_learning)):
    for k in range(len(dataset_learning[recording][0])):
        single_event = [dataset_learning[recording][0][k], dataset_learning[recording][1][k]]
        time_surface = Time_Surface_event(xdim = ts_size,
                                          ydim = ts_size,
                                          event = single_event,
                                          timecoeff = tau,
                                          dataset = dataset_learning[recording],
                                          num_polarities = polarities,
                                          verbose = False)

        x = single_event[1][0]
        y = single_event[1][1]
        index = x + rec_size * y # 2D to 1D mapping
        ts[recording][index] = time_surface

# setting up the testing dataset
number_of_samples = len(dataset_testing)
ts_test = np.zeros((number_of_samples,number_of_features,ts_size,ts_size))
for recording in range(len(dataset_testing)):
    for k in range(len(dataset_testing[recording][0])):
        single_event = [dataset_testing[recording][0][k], dataset_testing[recording][1][k]]

        time_surface = Time_Surface_event(xdim = ts_size,
                                          ydim = ts_size,
                                          event = single_event,
                                          timecoeff = tau,
                                          dataset = dataset_testing[recording],
                                          num_polarities = polarities,
                                          verbose = False)

        x = single_event[1][0]
        y = single_event[1][1]
        index = x + rec_size * y # 2D to 1D mapping
        ts_test[recording][index] = time_surface

#The code takes in data formated as Number of Samples x Number of Features so we need to reshape
ts = ts.reshape((ts.shape[0]*ts.shape[1],-1))
ts_test = ts_test.reshape((ts_test.shape[0]*ts_test.shape[1],-1))

if learning:
    # Dimensionality of the model
    H=100     # let's start with 100
    D=ts_size**2    # dimensionality of observed data

    # Approximation parameters for Expectation Truncation (It has to be Hprime>=gamma)
    Hprime = 8
    gamma = 5
     
    # Import and instantiate a model
    from prosper.em.camodels.bsc_et import BSC_ET
    model = BSC_ET(D, H, Hprime, gamma)



    # Choose annealing schedule
    from prosper.em.annealing import LinearAnnealing
    anneal = LinearAnnealing(20) # decrease
    anneal['T'] = [(0, 5.), (.8, 1.)]
    anneal['Ncut_factor'] = [(0,0.),(2./3,1.)]
    anneal['anneal_prior'] = False


    my_data={'y':ts}
    model_params = model.standard_init(my_data)
    print ("model defined") 
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params    
    em.run()
    print ("em finished") 

    my_test_data={'y':ts_test}
    res=model.inference(anneal,em.lparams, my_test_data)
    sparse_codes = res['s'][:,0,:]#should be Number of samples x H
