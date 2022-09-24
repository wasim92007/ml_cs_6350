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

    ## Attributes::         Outlook,Temperature,Humidity,Wind,Play
    incomplete_datapoint = ['Missing','Mild','Normal','Weak','Yes']

    #### Strategy to fill the Missing data
    ## Use the most common value among the training instances with the same
    ## label, namely, their attribute ”Play” is ”Yes”

    ## Split the dataset where label is 'Yes'
    train_ds_yes = split_based_on_attrib(dataset=train_ds, attrib='Play', val='Yes', attrib_name_col_dic=attrib_name_col_dic)
    
    ## Get the most common value for the missing feature from the sub dataset with similar label
    most_common_value = get_majority_label(train_ds_yes[:,attrib_name_col_dic['Outlook']])
    print(f'Most common value for the outlook in the sub dataset is {most_common_value}')

    ## Interpolate the incomplete datapoint using the strategy
    interpolated_datapoint = [most_common_value, 'Mild','Normal','Weak','Yes']

    appended_train_ds = np.vstack((train_ds, interpolated_datapoint))
    print(f'appended_train_ds:{appended_train_ds}')

    for entropy_func in ['entropy', 'majority_error', 'gini_index']:
        max_attrib, max_gain = get_attribute_for_max_info_gain(dataset=appended_train_ds, attrib_name_col_dic=attrib_name_col_dic, attrib_name_vals_dic=attrib_name_vals_dic, return_pair=True, entropy_func=entropy_func)
        print(f'entropy_func:{entropy_func}, max_attrib:{max_attrib}, max_gain:{max_gain}')


