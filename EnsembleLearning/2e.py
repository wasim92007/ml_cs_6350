## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
from decision_tree import *
from random_forest import RandomForest
from visualization import *

if __name__ == '__main__':
    ## Hyper parameters
    num_random_forest_tree = 100
    #num_random_forest_tree = 5
    num_rounds  = 500
    #num_rounds  = 2
    max_depth   = float('inf')
    pos_label='yes'
    neg_label='no'
    pos_bin = 1
    neg_bin = -1
    #max_depth   = 2
    ## Specify the filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'dir_path:{dir_path}')
    train_ds_fname = 'data/bank/train.csv'
    train_ds_fname = os.path.join(dir_path, train_ds_fname)
    test_ds_fname  = 'data/bank/test.csv'
    test_ds_fname = os.path.join(dir_path, test_ds_fname)

    ## Parse the train dataset
    ori_train_ds = open_csv_file(filename=train_ds_fname)

    ## Information about the train dataset
    print(f'ori_train_ds shape:{ori_train_ds.shape}')

    ## Parse the test dataset
    ori_test_ds = open_csv_file(filename=test_ds_fname)

    ## Information about the test dataset
    print(f'ori_test_ds shape:{ori_test_ds.shape}')

    ## Create a attribute name: col value dictionary
    attrib_name_col_dic = {
        'age'       :0,
        'job'       :1,
        'marital'   :2,
        'education' :3,
        'default'   :4,
        'balance'   :5,
        'housing'   :6,
        'loan'      :7,
        'contact'   :8,
        'day'       :9,
        'month'     :10,
        'duration'  :11,
        'campaign'  :12,
        'pdays'     :13,
        'previous'  :14,
        'poutcome'  :15,
        'y'         :16
        }
    #print(f'attrib_name_col_dic:{attrib_name_col_dic}')

    col_attrib_name_dic = {}
    for attrib_name, col in attrib_name_col_dic.items():
        col_attrib_name_dic[col] = attrib_name

    ## Create an attribute name:[attribute values] dictionary
    attrib_name_vals_dic ={
        'age'       :"Numeric",
        'job'       :["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
        'marital'   :["married","divorced","single"],
        'education' :["unknown","secondary","primary","tertiary"],
        'default'   :["yes","no"],
        'balance'   :"Numeric",
        'housing'   :["yes","no"],
        'loan'      :["yes","no"],
        'contact'   :["unknown","telephone","cellular"],
        'day'       :"Numeric",
        'month'     :["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        'duration'  :"Numeric",
        'campaign'  :"Numeric",
        'pdays'     :"Numeric",
        'previous'  :"Numeric",
        'poutcome'  :["unknown","other","failure","success"],
    }
    #print(f'attrib_name_vals_dic:{attrib_name_vals_dic}')

    numeric_col_list = [attrib_name_col_dic[attrib] for attrib, val in attrib_name_vals_dic.items() if val == "Numeric" ]
    #print(f'Numeric column list:{numeric_col_list}')

    ## Convert numerics in integers
    train_ds = ori_train_ds
    #train_ds = convert_to_numeric(dataset=ori_train_ds, numeric_col_list=numeric_col_list)
    test_ds = ori_test_ds
    #test_ds = convert_to_numeric(dataset=ori_test_ds, numeric_col_list=numeric_col_list)
    #print(ori_test_ds[:5])
    #print(test_ds[:5])
    num_samples = len(train_ds)
    #num_samples = 5

    ## Iterate over the number of Forest Tree to save memory
    ## We will store the result and perform calculation later
    single_pred_list = []
    random_forest_pred_list = []
    for n in range(num_random_forest_tree):
        ## Intialize Random Forest class
        random_forest_cs = RandomForest(max_depth=max_depth,
                             num_rounds=num_rounds
                            )

        ### Initialize the weights
        #initial_weights = np.array([1/len(train_ds)]*len(train_ds))

        ## Perform bagging
        random_forest_cs.perform_random_forest(train_ds=train_ds[:],
                                   test_ds=test_ds[:],
                                   num_samples=num_samples,
                                   numeric_col_list=numeric_col_list, 
                                   attrib_name_col_dic=attrib_name_col_dic, 
                                   attrib_name_vals_dic=attrib_name_vals_dic,
                                   pos_label='yes',
                                   neg_label='no',
                                   line=f'Random Forest {n+1}/{num_random_forest_tree}: '
                                  )

        ## Predict using first tree
        single_pred = random_forest_cs.classifiers[0]['classifier'].predict_dataset_numeric(test_ds[:])

        ## Convert first tree prediction to binary
        single_pred_bin = map_labels_to_bins(single_pred[:,-1], pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)

        ## Add to list
        single_pred_list.append(single_pred_bin)

        
        ## Predict using Random Forest
        random_forest_pred, _ = random_forest_cs.predict_dataset(test_ds[:])

        ## Convert Random Forest prediction to binary
        random_forest_pred_bin = map_labels_to_bins(random_forest_pred[:,-1], pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)

        ## Add to list
        random_forest_pred_list.append(random_forest_pred_bin)

    ## Processing
    ## Convert ground truth to binary
    gt_bin = map_labels_to_bins(test_ds[:,-1], pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)
    
    ## Transpose
    single_pred_list        = np.transpose(single_pred_list)
    random_forest_pred_list = np.transpose(random_forest_pred_list)

    ## Average prediction
    single_pred_bin_avg = np.sum(single_pred_list, axis=1)/num_random_forest_tree
    random_forest_pred_bin_avg = np.sum(random_forest_pred_list, axis=1)/num_random_forest_tree

    #### Calculate bias
    ## For single random tree
    single_pred_bias = np.average((gt_bin - single_pred_bin_avg) ** 2)
    single_pred_var  = np.array([(gt_bin - single_pred_list[:,i]) ** 2 for i in range(single_pred_list.shape[1])]).sum(axis=1)/num_random_forest_tree
    single_pred_var = np.average(single_pred_var)
    single_pred_gse  = single_pred_bias + single_pred_var

    print(f'First random tree bias: {single_pred_bias}')
    print(f'First random tree variance: {single_pred_var}')
    print(f'First random tree gse: {single_pred_gse}')


    ## For Random Forest
    random_forest_pred_bias = np.average((gt_bin - random_forest_pred_bin_avg) ** 2)
    random_forest_pred_var  = np.array([(gt_bin - random_forest_pred_list[:,i]) ** 2 for i in range(random_forest_pred_list.shape[1])]).sum(axis=1)/num_random_forest_tree
    random_forest_pred_var = np.average(random_forest_pred_var)
    random_forest_pred_gse  = random_forest_pred_bias + random_forest_pred_var

    print(f'Random Forest bias: {random_forest_pred_bias}')
    print(f'Random Forest variance: {random_forest_pred_var}')
    print(f'Random Forest gse: {random_forest_pred_gse}')



