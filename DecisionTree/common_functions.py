## Import Libraries
import numpy as np
from scipy import stats

## Opens a csv file and return the parsed parameters as an array
def open_csv_file(filename, sep=','):
    with open(file=filename, mode='r') as f:
        dataset = []
        for line in f:
            cols = line.strip().split(sep)
            dataset.append(cols)
        dataset = np.array(dataset)
        return dataset

## Check if the labels are same
def are_all_labels_same(labels):
    return len(np.unique(labels)) == 1

## Return majority label
def get_majority_label(labels):
    return stats.mode(labels)[0][0]

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
    print(f'majority_label:{majority_label}, majority_count:{values_counts_dic[majority_label]}/{total}')
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
def split_based_on_attrib(attrib, val, attrib_name_col_dic, dataset, return_dataset=True):
    ## Get index corresponding to the attrib
    index = get_col_index(attrib, attrib_name_col_dic)

    ## Get a new dataset where the value of the attrib is the given value
    new_dataset = []
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
def get_information_gain(dataset, attrib, attrib_vals, attrib_name_col_dic, entropy_func='entropy'):
    ## Get index corresponding to the attrib
    index = get_col_index(attrib, attrib_name_col_dic)
    ## Get the initial Entropy entropy_func
    init_entropy = get_entropy_variant(labels=dataset[:,-1], entropy_func=entropy_func)
    print(f'init_entropy_{entropy_func}:{init_entropy}')
    ## Initial dataset size
    dataset_size = len(dataset)

    ## Implementing --
    ## Gain(S,A) = Entropy(S) - Sum(|S_v|/|S|Entropy(S_v))
    info_gain = init_entropy
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
        print(f'split info gain for {attrib} using {entropy_func}:{split_info_gain[attrib]}')
        if split_info_gain[attrib] > max_gain:
            max_gain   = split_info_gain[attrib]
            max_attrib = attrib
    print(f'split_info_gain:{split_info_gain}')
    if return_pair:
        return max_attrib, max_gain
    else:
        return max_gain
