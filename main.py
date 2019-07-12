import prosper 
import numpy as np
from utils.Time_Surface_generators import Time_Surface_event, Time_Surface_all

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.bsc_et import BSC_ET

import os 



# Each datapoint is of D = size*size
size = 3
 
ts=np.random.randn(100,30,size,size) #let's pretend these are timesurfaces till we get it running
ts_test=np.random.randn(50,30,size,size) #let's pretend these are timesurfaces till we get it running

#The code takes in data formated as Number of Samples x Number of Features so we need to reshape
#Not sure of the timesurface format but we need to adapt it similarly
ts = ts.reshape((ts.shape[0]*ts.shape[1],-1))
ts_test = ts_test.reshape((ts_test.shape[0]*ts_test.shape[1],-1))
######3



N = ts.shape[0]

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
