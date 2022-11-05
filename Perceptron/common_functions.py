## Import Libraries
import os
import numpy as np

## Creates a directory if it does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

## Opens a csv file and return the parsed parameters as an array
def open_csv_file(filename, sep=',', numeric_val=False):
    with open(file=filename, mode='r') as f:
        dataset = []
        for line in f:
            cols = line.strip().split(sep)
            dataset.append(cols)

        if numeric_val:
            dataset = np.array(dataset, dtype=np.float32)
        else:
            dataset = np.array(dataset)

        return dataset

## Get prediction accuracy
def get_prediction_accuracy(predictions, labels, in_percentage=True, weights=None):
    #print(f'predictions:\n{predictions}')
    #print(f'labels:\n{labels}')
    if weights is None:
        weights = np.array([1]*len(predictions))
    #result = np.where(predictions==labels, 1, 0)
    #print(f'result:\n{result}')
    #n_correct = result.sum()
    n_correct = 0
    for pred, label, weight in zip(predictions, labels, weights):
        if pred == label:
            n_correct += weight
    n_total   = sum(weights)
    accuracy = n_correct/n_total
    if in_percentage:
        accuracy *= 100
    #print(f'n_correct:{n_correct}, n_total:{n_total}, accuracy:{accuracy}')
    return accuracy