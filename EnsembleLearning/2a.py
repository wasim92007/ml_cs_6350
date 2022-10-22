## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
from decision_tree import *
from adaboost import AdaBoost
from visualization import *

if __name__ == '__main__':
    ## Hyper parameters
    num_rounds = 500
    max_depth  = 1
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

    #### Unit tes of the modified common functions
    ## get_majority_labels
    #get_majority_label(labels=['a', 'c', 'c', 'unknown'], weights=[1/4, 1/4, 1/4, 1/2], filter_unknown=True)
    #exit()

    #entropy = get_weighted_entropy(['a', 'a', 'a', 'b'], None)
    #print(entropy)

    ## Intialize Adaboost class
    adaboost_cs = AdaBoost(train_ds=train_ds,
                           test_ds=test_ds,
                           max_depth=max_depth,
                           num_rounds=num_rounds
                          )

    ## Initialize the weights
    initial_weights = np.array([1/len(train_ds)]*len(train_ds))

    ## Perform ada boosting
    adaboost_cs.perform_adaboost(dataset=train_ds,
                                 init_weights=initial_weights,
                                numeric_col_list=numeric_col_list, 
                                attrib_name_col_dic=attrib_name_col_dic, 
                                attrib_name_vals_dic=attrib_name_vals_dic,
                                pos_label='yes',
                                neg_label='no'
                                )

    
    ## Perform unit test
    train_prediction, train_prediction_hist = adaboost_cs.predict_datapoint(datapoint=train_ds[2], pos_label='yes', neg_label='no')
    #print(f'train_prediction:{train_prediction}')
    #print(f'train_prediction_hist:{train_prediction_hist}')

    ## Perform whole train_ds test
    train_predictions, train_predictions_hist = adaboost_cs.predict_dataset(dataset=train_ds, pos_label='yes', neg_label='no')
    #print(f'train_predictions[:5]:{predictions[:5]}')
    #print(f'train_predictions_hist[:5]:{train_predictions_hist[:5]}')

    train_predictions_hist_dic = {t:[] for t in range(adaboost_cs.num_rounds)}
    for train_pred_datapoint_hist in train_predictions_hist:
        for t in range(adaboost_cs.num_rounds):
            train_predictions_hist_dic[t].append(train_pred_datapoint_hist[t])

    #print(f'train_predictions_hist_dic[0][:5]:{train_predictions_hist_dic[0][:5]}')

    for t in range(adaboost_cs.num_rounds):
        t_train_acc = get_prediction_accuracy(predictions=train_predictions_hist_dic[t], labels=train_ds[:,-1], in_percentage=False)
        ada_eta_t_train = 1 - t_train_acc
        adaboost_cs.classifiers[t]['ada_eta_t_train'] = ada_eta_t_train
        #print(f'eta_t_train:{adaboost_cs.classifiers[t]["eta_t_train"]}')
        #print(f'ada_eta_t_train:{adaboost_cs.classifiers[t]["ada_eta_t_train"]}')

    ## Perform whole test_ds test
    test_predictions, test_predictions_hist = adaboost_cs.predict_dataset(dataset=test_ds, pos_label='yes', neg_label='no')
    #print(f'test_predictions[:5]:{predictions[:5]}')
    #print(f'test_predictions_hist[:5]:{test_predictions_hist[:5]}')

    test_predictions_hist_dic = {t:[] for t in range(adaboost_cs.num_rounds)}
    for test_pred_datapoint_hist in test_predictions_hist:
        for t in range(adaboost_cs.num_rounds):
            test_predictions_hist_dic[t].append(test_pred_datapoint_hist[t])

    #print(f'test_predictions_hist_dic[0][:5]:{test_predictions_hist_dic[0][:5]}')

    for t in range(adaboost_cs.num_rounds):
        t_test_acc = get_prediction_accuracy(predictions=test_predictions_hist_dic[t], labels=test_ds[:,-1], in_percentage=False)
        ada_eta_t_test = 1 - t_test_acc
        adaboost_cs.classifiers[t]['ada_eta_t_test'] = ada_eta_t_test
        #print(f'eta_t_test:{adaboost_cs.classifiers[t]["eta_t_test"]}')
        #print(f'ada_eta_t_test:{adaboost_cs.classifiers[t]["ada_eta_t_test"]}')
    

    ## Prepare the errors list
    alpha_ts = []
    classifer_train_errors = []
    classifer_test_errors  = []
    ada_train_errors = []
    ada_test_errors  = []
    for t in range(adaboost_cs.num_rounds):
        alpha_ts.append(adaboost_cs.classifiers[t]['alpha_t'])
        classifer_train_errors.append(adaboost_cs.classifiers[t]['eta_t_train'])
        classifer_test_errors.append (adaboost_cs.classifiers[t]['eta_t_test'])
        ada_train_errors.append(adaboost_cs.classifiers[t]['ada_eta_t_train'])
        ada_test_errors.append (adaboost_cs.classifiers[t]['ada_eta_t_test'])

    ## Plot the alpha_t variation with t
    print(f'Plotting alpha_t vs t graph')
    label_dic = {'xlabel': 'Iteration(t)', 'ylabel': 'alpha_t', 'legend':['train'], 'title':'Adaboost alpha_t'}
    plot_one_data_func(data_1=alpha_ts, graph_name='./results/2a_alpha_t.png', label_dic=label_dic)

    ## Plot the stump train test error variation with t
    print(f'Plotting stump error vs t graph')
    label_dic = {'xlabel': 'Iteration(t)', 'ylabel': 'Error(decision stump)', 'legend':['train', 'test'], 'title':'Adaboost decision stump error'}
    plot_two_data_func(data_1=classifer_train_errors, data_2=classifer_test_errors, graph_name='./results/2a_classifier_error.png', label_dic=label_dic)

    ## Plot the ensemble train test error variation with t
    print(f'Plotting ensemble error vs t graph')
    label_dic = {'xlabel': 'Iteration(t)', 'ylabel': 'Error(AdaBoost Ensemble)', 'legend':['train', 'test'], 'title':'Adaboost ensemble error'}
    plot_two_data_func(data_1=ada_train_errors, data_2=ada_test_errors, graph_name='./results/2a_ensemble_error.png', label_dic=label_dic)