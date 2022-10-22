## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
from decision_tree import *
from bagging import Bagging
from visualization import *

if __name__ == '__main__':
    ## Hyper parameters
    num_bagged_tree = 100
    #num_bagged_tree = 5
    num_rounds  = 500
    #num_rounds  = 10
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

    #### Unit tes of the modified common functions
    ## get_majority_labels
    #get_majority_label(labels=['a', 'c', 'c', 'unknown'], weights=[1/4, 1/4, 1/4, 1/2], filter_unknown=True)
    #exit()

    #entropy = get_weighted_entropy(['a', 'a', 'a', 'b'], None)
    #print(entropy)

    ## Collect bagged trees
    bagged_trees = []
    for n in range(num_bagged_tree):
        ## Intialize Bagging class
        bagging_cs = Bagging(max_depth=max_depth,
                             num_rounds=num_rounds
                            )

        ### Initialize the weights
        #initial_weights = np.array([1/len(train_ds)]*len(train_ds))

        ## Perform bagging
        bagging_cs.perform_bagging(train_ds=train_ds[:],
                                   test_ds=test_ds[:],
                                   num_samples=num_samples,
                                   numeric_col_list=numeric_col_list, 
                                   attrib_name_col_dic=attrib_name_col_dic, 
                                   attrib_name_vals_dic=attrib_name_vals_dic,
                                   pos_label='yes',
                                   neg_label='no',
                                   line=f'Bagged tree {n+1}/{num_bagged_tree}: '
                                  )

        ## Add bagged tree to collection
        bagged_trees.append(bagging_cs)

    ##  Checking variance
    single_tree_bias_list = []
    single_tree_var_list  = []
    single_tree_gse_list  = []
    bagged_tree_bias_list = []
    bagged_tree_var_list  = []
    bagged_tree_gse_list  = []
    for i, test_datapoint in enumerate(test_ds):
        print(f'Testing datapoint {i+1}/{len(test_ds)}')
        ## Get datapoint gt
        test_datapoint_gt = test_datapoint[-1]
        test_datapoint_gt_bin = map_label_to_bin(test_datapoint[-1], pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)
        ## final prediction on datapoint
        single_prediction_final = 0
        bagged_prediction_final = 0
        single_predictions = []
        bagged_predictions = []

        for n in range(num_bagged_tree):

            #### Single tree prediction
            ## Get a bagged tree
            bagging_cs = bagged_trees[0]

            ## Get bagged tree first tree prediction
            single_prediction = bagging_cs.classifiers[0]['classifier'].predict_datapoint_numeric(test_datapoint)

            ## Convert first tree prediction to binary
            single_prediction_bin = map_label_to_bin(single_prediction, pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)
            ## Add first tree prediction to list for calculation of variance later
            single_predictions.append(single_prediction_bin)

            ## Add to single tree final prediction
            single_prediction_final += single_prediction_bin


            #### Bagged tree prediction
            ## Get bagged tree prediction
            bagged_prediction = bagging_cs.predict_datapoint(test_datapoint)

            ## Convert bagged tree prediction to binary
            bagged_prediction_bin = map_label_to_bin(bagged_prediction, pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)
            ## Add bagged tree prediction to list for calculation of variance later
            bagged_predictions.append(bagged_prediction_bin)

            ## Add to single tree final prediction
            bagged_prediction_final += bagged_prediction_bin


        #### Get the average prediction

        ### Single tree average prediction
        ## Calculate the single tree prediction average
        avg_single_prediction = single_prediction_final/num_bagged_tree
        ## Convert the single tree prediction average to binary label
        #avg_single_prediction = map_bin_to_label(bin_label=avg_single_prediction>0,pos_label=1, neg_label=-1, pos_bin=True, neg_bin=False)
        ## Calculate the single tree bias
        single_prediction_bias = (test_datapoint_gt_bin - avg_single_prediction)**2
        ## Calculate the single tree bias
        single_prediction_var = [(avg_single_prediction - pred_bin)**2 for pred_bin in single_predictions]
        single_prediction_var = sum(single_prediction_var)/num_bagged_tree
        ## Calculate the single tree general squared error
        single_prediction_gse = single_prediction_bias + single_prediction_var

        ## Add single tree bias, var and gse to list
        single_tree_bias_list.append(single_prediction_bias)
        single_tree_var_list.append(single_prediction_var)
        single_tree_gse_list.append(single_prediction_gse)
    
        ### Bagged tree average prediction
        ## Calculate the bagged tree prediction average
        avg_bagged_prediction = bagged_prediction_final/num_bagged_tree
        ## Convert the bagged tree prediction average to binary label
        #avg_bagged_prediction = map_bin_to_label(bin_label=avg_bagged_prediction>0,pos_label=1, neg_label=-1, pos_bin=True, neg_bin=False)
        ## Calculate the bagged tree bias
        bagged_prediction_bias = (test_datapoint_gt_bin - avg_bagged_prediction)**2
        ## Calculate the bagged tree bias
        bagged_prediction_var = [(avg_bagged_prediction - pred_bin)**2 for pred_bin in bagged_predictions]
        bagged_prediction_var = sum(bagged_prediction_var)/num_bagged_tree
        ## Calculate the bagged tree general squared error
        bagged_prediction_gse = bagged_prediction_bias + bagged_prediction_var
    
        ## Add bagged tree bias, var and gse to list
        bagged_tree_bias_list.append(bagged_prediction_bias)
        bagged_tree_var_list.append (bagged_prediction_var)
        bagged_tree_gse_list.append (bagged_prediction_gse)

    print(f'Single tree bias:{sum(single_tree_bias_list)/len(test_ds)}')
    print(f'Single tree variance:{sum(single_tree_var_list)/len(test_ds)}')
    print(f'Single tree gse:{sum(single_tree_gse_list)/len(test_ds)}')

    print(f'Bagged tree bias:{sum(bagged_tree_bias_list)/len(test_ds)}')
    print(f'Bagged tree variance:{sum(bagged_tree_var_list)/len(test_ds)}')
    print(f'Bagged tree gse:{sum(bagged_tree_gse_list)/len(test_ds)}')