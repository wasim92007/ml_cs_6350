## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
from regression import *
from adaboost import AdaBoost
from visualization import *

if __name__ == '__main__':
    ## Hyper parameters
    num_rounds = 500
    max_depth  = 1
    ## Specify the filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'dir_path:{dir_path}')
    train_ds_fname = 'data/1_5_train.csv'
    train_ds_fname = os.path.join(dir_path, train_ds_fname)
    test_ds_fname  = 'data/1_5_test.csv'
    test_ds_fname = os.path.join(dir_path, test_ds_fname)

    ## Parse the train dataset
    ori_train_ds = open_csv_file(filename=train_ds_fname)

    ## Information about the train dataset
    print(f'ori_train_ds shape:{ori_train_ds.shape}')

    ## Parse the test dataset
    ori_test_ds = open_csv_file(filename=test_ds_fname)

    ## Information about the test dataset
    print(f'ori_test_ds shape:{ori_test_ds.shape}')

    ## Convert numerics in integers
    train_ds = ori_train_ds
    train_ds = convert_to_numeric(dataset=ori_train_ds, numeric_col_list=[i for i in range(len(train_ds[0]))])
    #print(train_ds[0])
    test_ds = ori_test_ds
    test_ds = convert_to_numeric(dataset=ori_test_ds, numeric_col_list=[i for i in range(len(test_ds[0]))])
    #print(ori_test_ds[:5])
    #print(test_ds[:5])

    ##### Hyperparameter
    lr         = 0.1
    n_epoch    = 1
    #batch_size = len(train_ds)
    batch_size = 1
    #w          = np.array([-1, 1, -1, 1])[np.newaxis,:]
    w          = np.array([0, 0, 0, 0])[np.newaxis,:]

    ## Initiliaze LMS Linear Regression Class
    lms_regressor = LMS_Linear_Regression(lr=lr,
                                          n_epoch=n_epoch,
                                          batch_size=batch_size
                                          )

    ## Initialize weights
    lms_regressor.initialize_w(w)

    ## Train
    lms_regressor.train(train_ds=train_ds)