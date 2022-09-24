## External library import
import numpy as np
import os

## Internal library import
from common_functions import *
from decision_tree import *


if __name__ == '__main__':
    ## Specify the filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'dir_path:{dir_path}')
    train_ds_fname = 'data/car/train.csv'
    train_ds_fname = os.path.join(dir_path, train_ds_fname)
    test_ds_fname  = 'data/car/test.csv'
    test_ds_fname = os.path.join(dir_path, test_ds_fname)


    ## Parse the train dataset
    train_ds = open_csv_file(filename=train_ds_fname)

    ## Information about the train dataset
    print(f'train_ds shape:{train_ds.shape}')

    ## Parse the test dataset
    test_ds = open_csv_file(filename=test_ds_fname)

    ## Information about the test dataset
    print(f'test_ds shape:{test_ds.shape}')

    ## Create a attribute name: col value dictionary
    attrib_name_col_dic = {
        'buying'  :0,
        'maint'   :1,
        'doors'   :2,
        'persons' :3,
        'lug_boot':4,
        'safety'  :5,
        'label'   :6
        }
    #print(f'attrib_name_col_dic:{attrib_name_col_dic}')

    ## Create an attribute name:[attribute values] dictionary
    attrib_name_vals_dic ={
        'buying'  :['vhigh', 'high', 'med', 'low'],
        'maint'   :['vhigh', 'high', 'med', 'low'],
        'doors'   :['2', '3', '4', '5more'],
        'persons' :['2', '4', 'more'],
        'lug_boot':['small', 'med', 'big'],
        'safety'  :['low', 'med', 'high']
    }
    #print(f'attrib_name_vals_dic:{attrib_name_vals_dic}')

    ## Hyper parameters
    MAX_DEPTH = 6
    entropy_funcs = ['entropy', 'majority_error', 'gini_index']
    #entropy_funcs = ['entropy']
    #entropy_funcs = ['majority_error']
    #entropy_funcs = ['gini_index']

    ## Train and test on both training as we as test data
    for max_depth in range(1,MAX_DEPTH+1):
        print(f'\nIterating for max_depth:{max_depth}')

        for entropy_func in entropy_funcs:
            print(f'\nUsing {entropy_func} for attribute selection')

            ## Instantiate Car Decision Tree Class
            car_dtree_cs = Decision_Tree(
                max_depth=max_depth,
                entropy_func=entropy_func
                )

            ## Build the decision tree
            car_dtree_cs.build_tree(
                dataset=train_ds,
                attrib_name_col_dic=attrib_name_col_dic,
                attrib_name_vals_dic=attrib_name_vals_dic
                )

            ### Print the decision tree
            #print(f'\nDecision Tree Learned')
            #car_dtree_cs.print_tree(dfs=False)

            ## Predict on train dataset
            train_predictions = car_dtree_cs.predict_dataset(train_ds)
            ## Calculate the train accuracy
            train_accuracy = get_prediction_accuracy(train_predictions[:,-1], labels=train_ds[:,-1])
            print(f'Train dataset average accuracy:{train_accuracy}, error:{100-train_accuracy}')

            ## Predict on test dataset
            test_predictions = car_dtree_cs.predict_dataset(test_ds)
            ## Calculate the test accuracy
            test_accuracy = get_prediction_accuracy(test_predictions[:,-1], labels=test_ds[:,-1])
            print(f'Test dataset average accuracy:{test_accuracy}, error:{100-test_accuracy}')