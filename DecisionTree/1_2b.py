## External library import
import os
import numpy as np

## Internal library import
from common_functions import *
from decision_tree import *


if __name__ == '__main__':
    ## Specify the filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'dir_path:{dir_path}')

    ## Get the training dataset
    train_ds_fname = './data/1_2_train.csv'
    train_ds_fname = os.path.join(dir_path, train_ds_fname)

    test_ds_fname = './data/1_2_test.csv'
    test_ds_fname = os.path.join(dir_path, test_ds_fname)

    ## Parse the training dataset
    ori_train_ds = open_csv_file(filename=train_ds_fname)

    ## Parse the test dataset
    ori_test_ds = open_csv_file(filename=test_ds_fname)

    ## Information about the training dataset
    #print(f'train_ds shape:{ori_train_ds.shape}')

    ## Create a attribute name: col value dictionary
    attrib_name_col_dic = {}
    for i, attrib in enumerate(ori_train_ds[0,:]):
        attrib_name_col_dic[attrib] = i
    #print(f'attrib_name_col_dic:{attrib_name_col_dic}')

    ## Creating a train dataset without the header
    train_ds = ori_train_ds[1:,:]
    #print(f'train_ds:{train_ds}')

    ## Creating a test dataset without the header
    test_ds = ori_test_ds[1:,:]
    #print(f'test_ds:{test_ds}')

    ## Split into feature and label array
    features = train_ds[:, :-1]
    #print(f'features: {features}')
    labels   = train_ds[:, -1]
    #print(f'labels: {labels}')

    ## Intial entropy of the dataset
    #E0 = get_entropy(labels)
    #print(f'Initial entropy E0: {E0}')

    ## Create an attribute name:[attribute values] dictionary
    attrib_name_vals_dic ={}
    for attrib, col in attrib_name_col_dic.items():
        if attrib != 'Play':
            attrib_name_vals_dic[attrib] = np.unique(train_ds[:,col])
            if col == 2:
                attrib_name_vals_dic[attrib] = np.append(attrib_name_vals_dic[attrib],'Low')
    #print(f'attrib_name_vals_dic:{attrib_name_vals_dic}')

    ## Decision Tree for car dataset
    ## Hyper-parameters
    max_depth = 6
    #entropy_func   = 'entropy'
    #entropy_func   = 'majority_error'
    entropy_func   = 'gini_index'
    bin_dtree_cs = Decision_Tree(max_depth=max_depth, entropy_func=entropy_func)

    ## Build the decision tree
    bin_dtree_cs.build_tree(dataset=train_ds, attrib_name_col_dic=attrib_name_col_dic, attrib_name_vals_dic=attrib_name_vals_dic)

    ## Print the decision tree
    bin_dtree_cs.print_tree()

    '''
    ## Predict on a test datapoint
    for test_datapoint in test_ds:
        prediction = bin_dtree_cs.predict_datapoint(datapoint=test_datapoint[:bin_dtree_cs.num_features])
        print(f'test_datapoint:{test_datapoint}, prediction:{prediction}')
    '''

    ## Predict on a test dataset
    predictions = bin_dtree_cs.predict_dataset(test_ds)
    print(f'predictions:\n{predictions}')
