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
    num_rounds = 20
    ## Specify the filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'dir_path:{dir_path}')
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

    ## Intialize Adaboost class
    adaboost_cs = AdaBoost(train_ds=train_ds, test_ds=test_ds, num_rounds=num_rounds)

    ## Initialize the weights
    initial_weights = np.array([1/len(train_ds)]*len(train_ds))

    ## Perform ada boosting
    adaboost_cs.perform_adaboost(dataset=train_ds, init_weights=initial_weights, attrib_name_col_dic=attrib_name_col_dic, attrib_name_vals_dic=attrib_name_vals_dic, pos_label='Yes', neg_label='No')

    ## Perform unit test
    train_prediction, train_prediction_hist = adaboost_cs.predict_datapoint(datapoint=train_ds[5], pos_label='Yes', neg_label='No')
    #print(f'train_prediction:{train_prediction}')
    #print(f'train_prediction_hist:{train_prediction_hist}')

    ## Perform whole train_ds test
    train_predictions, train_predictions_hist = adaboost_cs.predict_dataset(dataset=train_ds, pos_label='Yes', neg_label='No')
    #print(f'train_predictions[:5]:{predictions[:5]}')
    #print(f'train_predictions_hist[:5]:{train_predictions_hist[:]}')

    train_predictions_hist_dic = {t:[] for t in range(adaboost_cs.num_rounds)}
    for train_pred_datapoint_hist in train_predictions_hist:
        for t in range(adaboost_cs.num_rounds):
            train_predictions_hist_dic[t].append(train_pred_datapoint_hist[t])

    #print(f'train_predictions_hist_dic[0][:5]:{train_predictions_hist_dic[0][:5]}')

    for t in range(adaboost_cs.num_rounds):
        t_train_acc = get_prediction_accuracy(predictions=train_predictions_hist_dic[t], labels=test_ds[:,-1], in_percentage=False)
        ada_eta_t_train = 1 - t_train_acc
        adaboost_cs.classifiers[t]['ada_eta_t_train'] = ada_eta_t_train
        #print(f'eta_t_train:{adaboost_cs.classifiers[t]["eta_t_train"]}')
        print(f'ada_eta_t_train:{adaboost_cs.classifiers[t]["ada_eta_t_train"]}')

    ## Perform whole test_ds test
    test_predictions, test_predictions_hist = adaboost_cs.predict_dataset(dataset=test_ds, pos_label='Yes', neg_label='No')
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
        print(f'ada_eta_t_test:{adaboost_cs.classifiers[t]["ada_eta_t_test"]}')
    

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
    label_dic = {'xlabel': 'Iteration(t)', 'ylabel': 'alpha_t', 'legend':['train'], 'title':'Adaboost alpha_t'}
    plot_one_data_func(data_1=alpha_ts, graph_name='./results/2a_alpha_t.png', label_dic=label_dic)

    ## Plot the stump train test error variation with t
    label_dic = {'xlabel': 'Iteration(t)', 'ylabel': 'Error(decision stump)', 'legend':['train', 'test'], 'title':'Adaboost decision stump error'}
    plot_two_data_func(data_1=classifer_train_errors, data_2=classifer_test_errors, graph_name='./results/2a_classifier_error.png', label_dic=label_dic)

    ## Plot the ensemble train test error variation with t
    label_dic = {'xlabel': 'Iteration(t)', 'ylabel': 'Error(AdaBoost Ensemble)', 'legend':['train', 'test'], 'title':'Adaboost ensemble error'}
    plot_two_data_func(data_1=ada_train_errors, data_2=ada_test_errors, graph_name='./results/2a_ensemble_error.png', label_dic=label_dic)