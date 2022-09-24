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

    '''
    ## Split dataset based on numeric value
    for attrib, attrib_vals in attrib_name_vals_dic.items():
        if attrib_vals == "Numeric":
            first_half_train_ds = split_based_on_attrib(attrib=attrib, val="Numeric", attrib_name_col_dic=attrib_name_col_dic, dataset=train_ds[:10], first_half=False)
            print(first_half_train_ds)
            exit()
    '''

    '''
    for attrib, attrib_vals in attrib_name_vals_dic.items():
        if attrib_vals == "Numeric":
            info_gain = get_information_gain(dataset=train_ds, attrib=attrib, attrib_vals=attrib_vals, attrib_name_col_dic=attrib_name_col_dic)
            print(info_gain)
            exit()
    '''

    '''
    get_attribute_for_max_info_gain(dataset=train_ds, attrib_name_vals_dic=attrib_name_vals_dic, attrib_name_col_dic=attrib_name_col_dic)
    '''
    

    ## Hyper parameters
    MAX_DEPTH = 16
    entropy_funcs = ['entropy', 'majority_error', 'gini_index']
    #entropy_funcs = ['entropy']
    #entropy_funcs = ['majority_error']
    #entropy_funcs = ['gini_index']

    ## Train and test on both training as we as test data
    for entropy_func in entropy_funcs:
        print(f'\nUsing {entropy_func} for attribute selection')

        for max_depth in range(1,MAX_DEPTH+1):
            print(f'\nIterating for max_depth:{max_depth}')

            ## Instantiate Bank Decision Tree Class
            bank_dtree_cs = Decision_Tree(
                max_depth=max_depth,
                entropy_func=entropy_func
                )

            ## Set the numeric column list
            bank_dtree_cs.set_numeric_col_list(numeric_col_list)

            ## Build the decision tree
            bank_dtree_cs.build_tree(
                dataset=train_ds,
                attrib_name_col_dic=attrib_name_col_dic,
                attrib_name_vals_dic=attrib_name_vals_dic
                )

            ### Print the decision tree
            #print(f'\nDecision Tree Learned')
            #bank_dtree_cs.print_tree(dfs=False)

            '''
            prediction = bank_dtree_cs.predict_datapoint_numeric(train_ds[0])
            print(f'{train_ds[0]}, prediction:{prediction}')
            '''

            ## Predict on train dataset
            train_predictions = bank_dtree_cs.predict_dataset_numeric(train_ds)
            ## Calculate the train accuracy
            train_accuracy = get_prediction_accuracy(train_predictions[:,-1], labels=train_ds[:,-1])
            print(f'Train dataset average accuracy:{train_accuracy}, error:{100-train_accuracy}')

            ## Predict on test dataset
            test_predictions = bank_dtree_cs.predict_dataset_numeric(test_ds)
            ## Calculate the test accuracy
            test_accuracy = get_prediction_accuracy(test_predictions[:,-1], labels=test_ds[:,-1])
            print(f'Test dataset average accuracy:{test_accuracy}, error:{100-test_accuracy}')