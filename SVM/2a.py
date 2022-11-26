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

        ## Initalize learning rate and a
        lr_init, a = 1, 1
        lr_scheduler = lambda epoch : lr_init / (1 + (lr_init * epoch)/a)

        ## Initialize the primal svm
        primal_svm = Primal_SVM(lr_scheduler=lr_scheduler, c=c, max_epochs=MAX_EPOCHS)

        ## Perform training
        primal_svm.train(train_ds=train_ds)

        ## Perform test on training dataset
        train_predictions = primal_svm.test(test_ds=train_ds)
        train_accuracy    = np.mean(train_ds[:,-1] == train_predictions)

        ## Perform test on testing dataset
        test_predictions = primal_svm.test(test_ds=test_ds)
        test_accuracy    = np.mean(test_ds[:,-1] == test_predictions)

        print(f'Learned weight for c:{c} is {primal_svm.w[0][:-1]}')
        print(f'Learned bias for c:{c} is {primal_svm.w[0][-1]}')
        print(f'Training accuracy: {train_accuracy * 100}, error: {(1-train_accuracy)*100}')
        print(f'Testing accuracy: {test_accuracy * 100}, error: {(1-test_accuracy)*100}')
