## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
from svm import *

if __name__ == '__main__':
    pass
    ## Hyper parameters
    MAX_EPOCHS = 100
    C_LIST     = [100/873, 500/873, 700/873]

    ## Specify the filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'dir_path:{dir_path}')
    train_ds_fname = 'data/bank-note/train.csv'
    train_ds_fname = os.path.join(dir_path, train_ds_fname)
    test_ds_fname  = 'data/bank-note/test.csv'
    test_ds_fname = os.path.join(dir_path, test_ds_fname)

    ## Parse the train dataset
    ori_train_ds = open_csv_file(filename=train_ds_fname, numeric_val=True)

    ## Information about the train dataset
    print(f'ori_train_ds shape:{ori_train_ds.shape}')

    ## Parse the test dataset
    ori_test_ds = open_csv_file(filename=test_ds_fname, numeric_val=True)

    ## Information about the test dataset
    print(f'ori_test_ds shape:{ori_test_ds.shape}')

    ## Preprocess train and test dataset
    ## Convert labels to 1, -1
    train_ds = ori_train_ds
    train_ds[:,-1] = (train_ds[:,-1] - 0.5) * 2
    test_ds  = ori_test_ds
    test_ds[:,-1]  = (test_ds[:,-1]  - 0.5) * 2

    ## Train
    for c in C_LIST:
        print('=============================================================')
        print(f'Current C value:{c}')

        ## Initialize the primal svm
        dual_svm = Dual_SVM()

        ## Perform training
        dual_svm.train(train_ds=train_ds, c=c)

        ## Perform test on training dataset
        train_predictions = dual_svm.test(test_ds=train_ds)
        train_accuracy    = np.mean(train_ds[:,-1] == train_predictions)

        ## Perform test on testing dataset
        test_predictions = dual_svm.test(test_ds=test_ds)
        test_accuracy    = np.mean(test_ds[:,-1] == test_predictions)

        print(f'Learned w* for c:{c} is {dual_svm.w_star}')
        print(f'Learned b* for c:{c} is {dual_svm.b_star}')
        print(f'Total support vectors count: {len(dual_svm.support_vectors)}')
        print(f'Training accuracy: {train_accuracy * 100}, error: {(1-train_accuracy)*100}')
        print(f'Testing accuracy: {test_accuracy * 100}, error: {(1-test_accuracy)*100}')
