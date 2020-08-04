import os
import h5py
import numpy as np
from readwriteatis_kaerdat import read_dataset

f = h5py.File('nmnist.h5', 'w')
train = f.create_group("train")
test = f.create_group("test")

data_path = "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/N-MNIST/"

train_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_path+"Train") for f in fn]
test_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_path+"Test") for f in fn]

for file_name in train_files:
    if file_name.endswith('.bin'):
        # read data
        timestamps, x, y, p = read_dataset(file_name)

        file_class = file_name.split('/')
        labels = [int(file_class[-2]),] * len(timestamps)
        name = file_class[-1].split('.')[0]+'_'+file_class[-2]

        # create dataset
        d = np.column_stack([timestamps, x, y, p, labels])
        train.create_dataset(name, data=d, dtype='f')

for file_name in test_files:
    if file_name.endswith('.bin'):
        # read data
        timestamps, x, y, p = read_dataset(file_name)

        file_class = file_name.split('/')
        labels = [int(file_class[-2]),] * len(timestamps)
        name = file_class[-1].split('.')[0]+'_'+file_class[-2]

        # create dataset
        d = np.column_stack([timestamps, x, y, p, labels])
        test.create_dataset(name, data=d, dtype='f')
