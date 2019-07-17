import prosper 
import numpy as np

from utils.Cards_loader import Cards_loader
from utils.Time_Surface_generators import Time_Surface_all

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.bsc_et import BSC_ET
import os 

# Decide whether to run the sparse coding algorithm
learning = False

# Each datapoint is of D = size*size
size = 35

# Importing the pips dataset for testing purposes
learning_set_length = 12
testing_set_length = 5

data_folder = "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/pips/"
dataset_learning, labels_learning, dataset_testing, labels_testing = Cards_loader(data_folder, learning_set_length,testing_set_length)

# Time surface of a full dataset 
tss = []
for i in range(learning_set_length):
    tss.append(Time_Surface_all(xdim=size, ydim=size, timestamp=dataset_learning[i][0][-1], timecoeff=1000, dataset=dataset_learning[i], num_polarities=1, minv=0.1, verbose=True))

ts_test = []
for i in range(testing_set_length):
    ts_test.append([Time_Surface_all(xdim=size, ydim=size, timestamp=dataset_testing[i][0][-1], timecoeff=1000, dataset=dataset_testing[i], num_polarities=1, minv=0.1, verbose=False)])

ts=np.random.randn(10,3,size,size) #let's pretend these are timesurfaces till we get it running
# print(ts)
# ts_test=np.random.randn(50,30,size,size) #let's pretend these are timesurfaces till we get it running

#The code takes in data formated as Number of Samples x Number of Features so we need to reshape
#Not sure of the timesurface format but we need to adapt it similarly
ts = ts.reshape((ts.shape[0]*ts.shape[1],-1))
# ts_test = ts_test.reshape((ts_test.shape[0]*ts_test.shape[1],-1))

if learning:
    # Dimensionality of the model
    H=100     # let's start with 100
    D=size**2    # dimensionality of observed data

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
