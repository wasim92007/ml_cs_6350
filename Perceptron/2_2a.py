## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
from perceptron import *

if __name__ == '__main__':
    ## Hyper parameters
    num_epochs = 10

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
    train_ds = ori_train_ds
    train_ds[:,-1] = (train_ds[:,-1] - 0.5) * 2
    test_ds  = ori_test_ds
    test_ds[:,-1]  = (test_ds[:,-1] - 0.5) * 2

    #### Run perceptron algorithm
    ## Initialize the weight to 0
    init_w = np.zeros((1, len(train_ds[0])))
    #init_w = np.random.rand(1, len(train_ds[0]))
    #init_w = np.ones((1, len(train_ds[0])))

    ## Initialize the Standard perceptron
    std_perceptron = Standard_Perceptron(num_epochs=num_epochs)

    ## Train the standard perceptron
    final_w = std_perceptron.train(train_ds=train_ds, init_w=init_w)

    ## Print learned weight
    print(f'Final learned weight:{final_w}')

    ## Predict on the test dataset
    predictions = std_perceptron.test(test_ds=test_ds)

    ## Calculate average prediction error
    test_acc = get_prediction_accuracy(predictions=predictions[:,-1], labels=test_ds[:,-1])

    ## Get error in percentage
    print(f'Test error percetage:{100-test_acc}')
