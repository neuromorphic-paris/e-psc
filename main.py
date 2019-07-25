import sys
import prosper 
import numpy as np
from scipy.stats import truncnorm

from utils.Cards_loader import Cards_loader
from utils.Time_Surface_generators import Time_Surface_all, Time_Surface_event

from prosper.em import EM
from prosper.em.annealing import LinearAnnealing
from prosper.em.camodels.bsc_et import BSC_ET
from prosper.utils import create_output_path 
from prosper.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt


import os 

np.set_printoptions(threshold=sys.maxsize)
output_path = create_output_path()

#### PARAMETERS ####

learning = True # Decide whether to run the sparse coding algorithm
classification	= True # Run classification

ts_size = 11 # size of the time surfaces
rec_size = 35 # size of the pip cards (square so dimension D = rec_size * rec_size)
tau = 5000 # time constant for the construction of time surfaces
polarities = 1 # number of polarities that we will use in the dataset (1 because polarities are not informative in the cards dataset)
feature_type = 1

#### BUILDING THE LEARNING DATASET ####
learning_set_length = 12
testing_set_length = 5

data_folder = "datasets/pips/"
dataset_learning, labels_learning, dataset_testing, labels_testing = Cards_loader(data_folder, learning_set_length,testing_set_length)

if feature_type == 0:
	# number of features  
	number_of_features = rec_size**2

	# setting up the learning dataset
	number_of_samples = len(dataset_learning)
	ts = np.ones((number_of_samples,number_of_features,ts_size,ts_size)) * truncnorm.rvs(0, 1e-6)
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

	# reshaping the time surface list as Number of Samples x Number of Features
	ts = ts.reshape((ts.shape[0]*ts.shape[1],-1))

	# setting up the testing dataset
	number_of_samples = len(dataset_testing)
	ts_test = np.ones((number_of_samples,number_of_features,ts_size,ts_size)) * truncnorm.rvs(0, 1e-6)
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

	# reshaping the time surface list as Number of Samples x Number of Features
	ts_test = ts_test.reshape((ts_test.shape[0]*ts_test.shape[1],-1))

elif feature_type == 1:
	# setting up the learning dataset
	sizes_of_training_samples = [len(dataset_learning[j][0]) for j in range(len(dataset_learning))]
	number_of_samples = sum(sizes_of_training_samples)

	number_of_features = ts_size**2
	ts = np.zeros((number_of_samples, ts_size, ts_size))

	idx = 0
	training_labels=[]
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
			ts[idx] = time_surface
			training_labels.append(recording)
			idx += 1
	ts = ts.reshape((ts.shape[0],-1))

	# setting up the testing dataset
	sizes_of_testing_samples = [len(dataset_testing[j][0]) for j in range(len(dataset_testing))]

	number_of_samples = sum(sizes_of_testing_samples)
	ts_test = np.zeros((number_of_samples, ts_size, ts_size))

	idx = 0
	test_labels=[]
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
			ts_test[idx] = time_surface
			test_labels.append(recording)
			idx += 1
	ts_test = ts_test.reshape((ts_test.shape[0],-1))

#### RUNNING THE SPARSE CODING ALGORITHM ####
if learning:
	# Dimensionality of the model
	H=100     # let's start with 100
	D=ts_size**2    # dimensionality of observed data

	# Approximation parameters for Expectation Truncation (It has to be Hprime>=gamma)
	Hprime = 8
	gamma = 6
	 
	# Import and instantiate a model
	from prosper.em.camodels.bsc_et import BSC_ET
	model = BSC_ET(D, H, Hprime, gamma)

	# Configure DataLogger
	print_list = ('T', 'L', 'pi', 'sigma')
	dlog.set_handler(print_list, TextPrinter) #prints things to terminal
	txt_list = ('T', 'L', 'pi', 'sigma')
	dlog.set_handler(txt_list, StoreToTxt, output_path+'/results.txt')  #stores things in a txt file
	h5_list = ('T', 'L', 'pi', 'sigma','W')
	dlog.set_handler(h5_list, StoreToH5, output_path+'/results.h5') #stores things in an h5 file

	# Choose annealing schedule
	from prosper.em.annealing import LinearAnnealing
	anneal = LinearAnnealing(120) # decrease
	anneal['T'] = [(0, 5.), (.8, 1.)]
	anneal['Ncut_factor'] = [(0,0.),(0.5,0.),(0.6,1.)]
	# anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
	# anneal['Ncut_factor'] = [(0,0.),(0.7,1.)]
	anneal['W_noise'] = [(0,np.std(ts)/2.),(0.7,0.)]
	# anneal['pi_noise'] = [(0,0.),(0.2,0.1),(0.7,0.)]
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
	dlog.close()

if classification:
	
	my_train_data={'y':ts}
	res_train = model.inference(anneal,em.lparams,my_train_data)
	
	train_features = []
	start = 0
	for i in range(len(sizes_of_training_samples)):
		stop = start + sizes_of_training_samples[i]
		train_features.append(res_train['s'][start:stop,0,:].mean(0))
		start = stop

	train_features=np.array(train_features)
	
	my_test_data={'y':ts_test}
	res_test = model.inference(anneal,em.lparams,my_test_data)

	test_features = []
	start = 0
	for i in range(len(sizes_of_testing_samples)):
		stop = start+sizes_of_testing_samples[i]
		test_features.append(res_test['s'][start:stop,0,:].mean(0))
		start = stop
	test_features=np.array(test_features)

	from sklearn.linear_model import LogisticRegression
	from sklearn import metrics

	lreg=LogisticRegression()
	lreg.fit(train_features,labels_learning)
	predicted_labels= lreg.predict(test_features)

	print("Classification report for classifier %s:\n%s\n"
	      % (lreg, metrics.classification_report(labels_testing, predicted_labels)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_testing, predicted_labels))
