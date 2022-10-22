## External library import
import os, sys
import numpy as np
from numpy import linalg as LA

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

    train_ds = np.array(train_ds)
    test_ds = np.array(test_ds)
    #print(ori_test_ds[:5])
    #print(test_ds[:5])

    ###### Hyperparameter for batch
    ##lr         = 0.00125
    ##lr         = 0.00025
    #lr         = 0.00005
    #n_epoch    = 100
    ##batch_size = len(train_ds)
    #batch_size = len(train_ds)
    #n_feat = len(train_ds[0])
    #w = np.ones((1, n_feat))

    ### Initiliaze LMS Linear Regression Class
    #lms_regressor = LMS_Linear_Regression(lr=lr,
    #                                      n_epoch=n_epoch,
    #                                      batch_size=batch_size
    #                                      )

    ### Initialize weights
    #lms_regressor.initialize_w(w)

    ### Train
    #train_loss = lms_regressor.train(train_ds=train_ds,
    #                                 till_convergence=True,
    #                                 tolerance=10e-7,
    #                    graph_name=f'./results/4a_loss_{str(lr)}.png'
    #)

    ### Get the weight vector
    #w_final_batch = lms_regressor.get_w()
    #print(f'Final learned weight:{w_final_batch}')

    ### Test
    #predictions, losses = lms_regressor.test_dataset(test_ds=test_ds)
    #print(f'Loss on test dataset:{losses}')

    ###### Hyperparameter for SGD
    ##lr         = 0.00125
    ##lr         = 0.00025
    #lr         = 0.00005
    ##lr         = 0.000001
    #n_epoch    = 100
    ##batch_size = len(train_ds)
    #batch_size = 1
    #n_feat = len(train_ds[0])
    #w = np.ones((1, n_feat))

    ### Initiliaze LMS Linear Regression Class
    #lms_regressor = LMS_Linear_Regression(lr=lr,
    #                                      n_epoch=n_epoch,
    #                                      batch_size=batch_size
    #                                      )

    ### Initialize weights
    #lms_regressor.initialize_w(w)

    ### Train
    #train_loss = lms_regressor.train(train_ds=train_ds,
    #                                 till_convergence=True,
    #                                 use_sgd=True,
    #                                 tolerance=10e-11,
    #                    graph_name=f'./results/4b_loss_{str(lr)}.png'
    #)

    ### Get the weight vector
    #w_final_sgd = lms_regressor.get_w()
    #print(f'Final learned weight SGD:{w_final_sgd}')

    ### Test
    #predictions, losses = lms_regressor.test_dataset(test_ds=test_ds)
    #print(f'Loss on test dataset using SGD:{losses}')

    ## Calculate optimal weight vector: w* = (X.XT)^(-1)XY
    ## Get the X
    X = train_ds[:,:-1] ## (53x7)

    ## Append 1 for bais term
    XT = np.hstack((X, np.ones((len(train_ds), 1)))) ## m x d = 53 x 8 
    X = np.transpose(XT) ## d x m =(8x53); m:= Num dp
    
    ## Get (X.XT)
    M = np.matmul(X, XT)   ## d x d = 8 x 8

    ## Get XY
    XY = np.matmul(X,train_ds[:,-1])[:,np.newaxis]    ## d x m x m x 1 = 8 x 1

    M_inv = LA.inv(M)
    #print(f'M_inv shape:{M_inv.shape}')

    w_optimum = M_inv @ XY
    print(f'Optimum weight:{w_optimum.T}')
    
    print(f'Final learned weight batch:{w_final_batch}')
    print(f'Final learned weight SGD:{w_final_sgd}')

    #print(f'Optimum weight:{w_optimum.T.shape}')
    #print(f'Batch weight:{w_final_batch.shape}')
    #print(f'SGD weight:{w_final_sgd.shape}')

    mse_opt_batch = np.mean(np.square(w_optimum.T - w_final_batch))
    mse_opt_sgd = np.mean(np.square(w_optimum.T - w_final_sgd))
    mse_batch_sgd = np.mean(np.square(w_final_batch - w_final_sgd))
    
    print(f'optimum-batch:{mse_opt_batch}')
    print(f'optimum-sgd:{mse_opt_sgd}')
    print(f'batch-sgd:{mse_batch_sgd}')


