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

