import os
import h5py
import numpy as np
from readwriteatis_kaerdat import readATIS_td

f = h5py.File('ncars.h5', 'w')
train = f.create_group("train")
test = f.create_group("test")

data_path = "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/n-CARS/"

train_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_path+"train") for f in fn]
test_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_path+"test") for f in fn]

for file_name in train_files:
    # read data
    data = readATIS_td(file_name, orig_at_zero = True, drop_negative_dt = True, verbose = False, events_restriction = [0, np.inf])

    # parse data into a compatible numpy array
    x = []; y = []
    timestamps = data[0]
    [x.append(coordinates[0]) for coordinates in data[1]]
    [y.append(coordinates[1]) for coordinates in data[1]]
    p = data[2]

    file_class = file_name.split('/')
    if file_class[-2] == 'cars':
        labels = [1,] * len(data[0])
        name = file_class[-1].split('.')[0]+'_car'
    elif file_class[-2] == 'background':
        labels = [0,] * len(data[0])
        name = file_class[-1].split('.')[0]+'_bgd'

    # create dataset
    d = np.column_stack([timestamps, x, y, p, labels])
    train.create_dataset(name, data=d, dtype='f')

for file_name in test_files:
    # read data
    data = readATIS_td(file_name, orig_at_zero = True, drop_negative_dt = True, verbose = False, events_restriction = [0, np.inf])

    # parse data into a compatible numpy array
    x = []; y = []
    timestamps = data[0]
    [x.append(coordinates[0]) for coordinates in data[1]]
    [y.append(coordinates[1]) for coordinates in data[1]]
    p = data[2]

    file_class = file_name.split('/')
    if file_class[-2] == 'cars':
        labels = [1,] * len(data[0])
        name = file_class[-1].split('.')[0]+'_car'
    elif file_class[-2] == 'background':
        labels = [0,] * len(data[0])
        name = file_class[-1].split('.')[0]+'_bgd'

    # create dataset
    d = np.column_stack([timestamps, x, y, p, labels])
    test.create_dataset(name, data=d, dtype='f')
