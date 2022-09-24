## Import Libraries
from ast import mod
import numpy as np
from scipy import stats
from statistics import median

## Opens a csv file and return the parsed parameters as an array
def open_csv_file(filename, sep=','):
    with open(file=filename, mode='r') as f:
        dataset = []
        for line in f:
            cols = line.strip().split(sep)
            dataset.append(cols)
        dataset = np.array(dataset)
        return dataset

## Convert Numeric values to numeric values
def convert_to_numeric(dataset, numeric_col_list):
    mod_dataset = []
    for datapoint in dataset:
        mod_datapoint = []
        for i, val in enumerate(datapoint):
            if i in numeric_col_list:
                val = int(val)
            mod_datapoint.append(val)
        mod_dataset.append(mod_datapoint)
    
    return mod_dataset

## Modify the dataset to replace the unkown
def replace_unknown_with_major(dataset, unknown_col_list, majority_attrib_val_dic):
    mod_dataset = []
    for datapoint in dataset:
        mod_datapoint = []
        for i, val in enumerate(datapoint):
            if val == "unknown" and i in unknown_col_list:
                val = majority_attrib_val_dic[i]
            mod_datapoint.append(val)
        mod_dataset.append(mod_datapoint)
        
    return np.array(mod_dataset)
## Check if the labels are same
def are_all_labels_same(labels):
    return len(np.unique(labels)) == 1

## Return majority label
def get_majority_label(labels, filter_unknown=False):
    if filter_unknown:
        mod_labels = list(filter(lambda x:x != "unknown", labels))
    else:
        mod_labels = labels
    return stats.mode(mod_labels)[0][0]

## Returns entropy
def get_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    p = {}
    entropy = 0
    for value, count in zip(values, counts):
        p[value] = count/total
        #print(f'p_{value}:{count}/{total}')
        if p[value] > 0:
            entropy -= p[value] * np.log2(p[value])
    return entropy

## Returns majority error
def get_majority_error(labels):
    values, counts = np.unique(labels, return_counts=True)
    values_counts_dic = dict(zip(values,counts))
    majority_label = get_majority_label(labels=labels)
    total = len(labels)
    #print(f'majority_label:{majority_label}, majority_count:{values_counts_dic[majority_label]}/{total}')
    majority_error = (total - values_counts_dic[majority_label])/total
    return majority_error

## Returns Gini Index (GI)
def get_gini_index(labels):
    values, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    p = {}
    gini_index = 1
    for value, count in zip(values, counts):
        p[value] = count/total
        #print(f'p_{value}:{count}/{total}')
        gini_index -= p[value]**2
    return gini_index

## Returns column index of a attrib
def get_col_index(attrib, attrib_name_col_dic):
    ## Check if the attrib is attrib or index
    if isinstance(attrib, str):
        index = attrib_name_col_dic[attrib]
    else:
        index = attrib
    return index

## Returns a dataset based on the attrib value
def split_based_on_attrib(attrib, val, attrib_name_col_dic, dataset, return_dataset=True, num_def_val="Numeric", first_half=True):
    ## Get index corresponding to the attrib
    index = get_col_index(attrib, attrib_name_col_dic)

    ## Get a new dataset where the value of the attrib is the given value
    new_dataset = []
    ## Handling of numeric value
    if val == num_def_val:
        ## Get median of the values
        median_val = median(map(int,dataset[:,index]))
        #print(f'median value:{median_val}')
        for i, datapoint in enumerate(dataset):
            if first_half:
                if int(datapoint[index]) <= median_val:
                    new_dataset.append(datapoint)
            else:
                if int(datapoint[index]) > median_val:
                    new_dataset.append(datapoint)
    ## Handling of non-numeric value
    else:
        for i, datapoint in enumerate(dataset):
            if datapoint[index] == val:
                new_dataset.append(datapoint)

    new_dataset = np.array(new_dataset)
    ## Return either dataset or features labels
    if return_dataset:
        return new_dataset
    else:
        return new_dataset[:,:-1], new_dataset[:, -1]

def get_entropy_variant(labels, entropy_func='entropy'):
    if entropy_func == 'entropy':
        return get_entropy(labels=labels)    
    elif entropy_func == 'majority_error':
        return get_majority_error(labels=labels)
    elif entropy_func == 'gini_index':
        return get_gini_index(labels=labels)
    else:
        return get_entropy(labels=labels)    

## Get information gain
def get_information_gain(dataset, attrib, attrib_vals, attrib_name_col_dic, entropy_func='entropy', num_def_val="Numeric"):
    ## Get index corresponding to the attrib
    index = get_col_index(attrib, attrib_name_col_dic)
    ## Get the initial Entropy entropy_func
    init_entropy = get_entropy_variant(labels=dataset[:,-1], entropy_func=entropy_func)
    #print(f'init_entropy_{entropy_func}:{init_entropy}')
    ## Initial dataset size
    dataset_size = len(dataset)

    ## Implementing --
    ## Gain(S,A) = Entropy(S) - Sum(|S_v|/|S|Entropy(S_v))
    info_gain = init_entropy

    ## Handling of numeric value
    if isinstance(attrib_vals, str) and attrib_vals == num_def_val:
        for val in [True, False]:
            sub_dataset = split_based_on_attrib(attrib=index, val=num_def_val, attrib_name_col_dic=attrib_name_col_dic, dataset=dataset, return_dataset=True, num_def_val=num_def_val, first_half=val)
            sub_dataset_size = len(sub_dataset)
            if len(sub_dataset) > 0:
                sub_dataset_frac = sub_dataset_size/dataset_size
                sub_dataset_entropy = get_entropy_variant(labels=sub_dataset[:,-1], entropy_func=entropy_func)
                #print(f'Entropy for {attrib}:{val} is {sub_dataset_entropy}')
                info_gain -= sub_dataset_frac*sub_dataset_entropy

    ## Handling of non-numeric value
    else:
        for val in attrib_vals:
            sub_dataset = split_based_on_attrib(attrib=index, val=val, attrib_name_col_dic=attrib_name_col_dic, dataset=dataset, return_dataset=True)
            sub_dataset_size = len(sub_dataset)
            if len(sub_dataset) > 0:
                sub_dataset_frac = sub_dataset_size/dataset_size
                sub_dataset_entropy = get_entropy_variant(labels=sub_dataset[:,-1], entropy_func=entropy_func)
                #print(f'Entropy for {attrib}:{val} is {sub_dataset_entropy}')
                info_gain -= sub_dataset_frac*sub_dataset_entropy
    
    return info_gain

def get_attribute_for_max_info_gain(dataset, attrib_name_vals_dic, attrib_name_col_dic, return_pair=False, entropy_func='entropy'):
    ## Check the infomation gain for intial split
    split_info_gain = {}
    max_attrib = ''
    max_gain   = -1
    for attrib, attrib_vals in attrib_name_vals_dic.items():
        split_info_gain[attrib] = get_information_gain(dataset=dataset, attrib=attrib, attrib_vals=attrib_vals, attrib_name_col_dic=attrib_name_col_dic, entropy_func=entropy_func)
        #print(f'split info gain for {attrib} using {entropy_func}:{split_info_gain[attrib]}')
        if split_info_gain[attrib] > max_gain:
            max_gain   = split_info_gain[attrib]
            max_attrib = attrib
    #print(f'split_info_gain:{split_info_gain}')
    if return_pair:
        return max_attrib, max_gain
    else:
        return max_gain

## Get prediction accuracy
def get_prediction_accuracy(predictions, labels, in_percentage=True):
    #print(f'predictions:\n{predictions}')
    #print(f'labels:\n{labels}')
    result = np.where(predictions==labels, 1, 0)
    #print(f'result:\n{result}')
    n_correct = result.sum()
    n_total   = len(labels)
    accuracy = n_correct/n_total
    if in_percentage:
        accuracy *= 100
    #print(f'n_correct:{n_correct}, n_total:{n_total}, accuracy:{accuracy}')
    return accuracy