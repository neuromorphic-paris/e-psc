import h5py
import numpy as np
from Cards_loader import Cards_loader
from readwriteatis_kaerdat import readATIS_td

f = h5py.File('pokerDVS.h5', 'w')
train = f.create_group("train")
test = f.create_group("test")

#### IMPORTING DATASET ####
learning_set_length = 12
testing_set_length = 5

data_folder = "../datasets/pips/"
dataset_learning, labels_learning, filenames_learning, dataset_testing, labels_testing, filenames_testing = Cards_loader(data_folder, learning_set_length, testing_set_length)

for i, data in enumerate(dataset_learning):
	x = []; y = []
	# get data
	timestamps = data[0]
	[x.append(coordinates[0]) for coordinates in data[1]]
	[y.append(coordinates[1]) for coordinates in data[1]]
	p = data[2]
	labels = [labels_learning[i],] * len(data[0])

	# build a numpy array from the data
	d = np.column_stack([timestamps, x, y, p, labels])
	train.create_dataset(filenames_learning[i], data=d, dtype='f')

for i, data in enumerate(dataset_testing):
	x = []; y = []
	# get data
	timestamps = data[0]
	[x.append(coordinates[0]) for coordinates in data[1]]
	[y.append(coordinates[1]) for coordinates in data[1]]
	p = data[2]
	labels = [labels_testing[i],] * len(data[0])

	# build a numpy array from the data
	d = np.column_stack([timestamps, x, y, p, labels])
	test.create_dataset(filenames_testing[i], data=d, dtype='f')