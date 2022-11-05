# ml_cs_6350
# Machine Learning CS 5350/6350 Fall 2022 University of Utah
This is a machine learning library developed by Wasim Akram Gazi for CS5350/6350 in University of Utah

# For Decision Tree
## Takes input:
### max_depth
Maximum depth of the Tree
### entropy function
Which entropy function to use
### numeric_col_list
List of column which contains numeric data
### num_def_val
Proxy value to represent numeric attibute type

## To train
Call the build tree function

## To test
Call predict_datapoint() or predict_dataset()

# For Adaboost, bagging and Random Forest
## Takes input
### AdaBoost
Training dataset, test dataset, maximum depth and number of rounds
### Bagging, Random Forest
Maximum depth and, number of rounds


## To train
Call the --
1. perform_adaboost, perform_bagging with training ds, weights, numeric column lsit, attribute name to column dictionary and positive negative lebel marker
2. perform_random_forest with train and test dataset, weights, numeric column lsit, attribute name to column dictionary and positive negative lebel marker, number of maximum feature subset

## To test
Call predict_datapoint() or predict_dataset()

# For LMS regression
## Takes input
learning rate, number of epochs, batch size

## To train
Call train() function with train dataset, till_converge (T/F), tolerance, use_sgd (T/F)

## To test
Call test_dataset()
