## External library import
import os, sys
import numpy as np
import random

## Internal library import
from common_functions import *
from decision_tree import *

class RandomForest(object):
    '''
    Class for Random Forest Algorithm
    '''
    def __init__(self,
                max_depth=float('inf'),
                num_rounds=2,
                classifier_type='decision_tree',
                entropy_func='weighted_entropy',
                *args, **kwargs) -> None:
        '''
        Initialize the Random Forest
        '''
        super().__init__()
        self.max_depth       = max_depth
        self.num_rounds      = num_rounds
        self.classifier_type = classifier_type
        self.entropy_func    = entropy_func
        self.classifiers      = {}

    def set_max_depth(self, max_depth):
        '''
        Sets the maximum depth
        '''
        self.max_depth = max_depth

    def get_max_depth(self):
        '''
        Returns maximum depth of the tree
        '''
        return self.max_depth

    def set_num_rounds(self, num_rounds):
        '''
        Sets the maximum rounds of Boosting
        '''
        self.num_rounds = num_rounds

    def get_num_rounds(self):
        '''
        Returns the maximum round of Boosting
        '''
        return self.num_rounds

    def perform_random_forest(self, train_ds, test_ds, t_init=0, *args, **kwargs):
        '''
        Perform Random Forest for t iterations
        '''
        #print(f'kwargs:{kwargs}'); input()
        pos_label = kwargs.get('pos_label', 'yes')
        neg_label = kwargs.get('neg_label', 'no')
        num_samples = kwargs.get('num_samples', len(train_ds))
        num_max_feat_subset=kwargs.get('num_max_feat_subset', 2)
        line = kwargs.get('line', '')
        for t in range(t_init, self.num_rounds):
            print(f'{line}Performing random forest build iteration {t+1}/{self.num_rounds}')

            ## Random sampling
            if self.classifier_type == 'decision_tree':
                ## Draw samples
                sampled_train_ds = []
                for i in range(num_samples):
                    rand_int = random.randint(0, len(train_ds)-1)
                    #print(f'random number:{rand_int}')
                    sampled_train_ds.append(train_ds[rand_int])
                sampled_train_ds = np.array(sampled_train_ds)

                ## Initialize the classifier
                classifier_cs = Decision_Tree(max_depth=self.max_depth, entropy_func=self.entropy_func)

                ## Set the numeric column list
                numeric_col_list = kwargs.get('numeric_col_list', [])
                classifier_cs.set_numeric_col_list(numeric_col_list)
            
                ## Build the decision tree
                classifier_cs.build_tree(dataset=sampled_train_ds[:], attrib_name_col_dic=kwargs['attrib_name_col_dic'], attrib_name_vals_dic=kwargs['attrib_name_vals_dic'], random_tree=True,num_max_feat_subset=num_max_feat_subset)

                ### Print the decision tree
                #print(f'\nDecision Tree Learned')
                #classifier_cs.print_tree(dfs=False)

                ## Using decision stump predict on the training set
                classifier_train_preds = classifier_cs.predict_dataset_numeric(sampled_train_ds[:])
                classifier_train_acc = get_prediction_accuracy(predictions=classifier_train_preds[:,-1], labels=sampled_train_ds[:,-1], in_percentage=False)
                #print(f'classifier train accuracy:{classifier_train_acc}')
                eta_t_train = 1 - classifier_train_acc

                ## Using decision stump predict on the training set
                classifier_test_preds = classifier_cs.predict_dataset_numeric(test_ds)
                classifier_test_acc = get_prediction_accuracy(predictions=classifier_test_preds[:,-1], labels=test_ds[:,-1], in_percentage=False)
                #print(f'classifier test accuracy:{classifier_test_acc}')
                eta_t_test = 1 - classifier_test_acc

                ## Add the decision tree to the learned algorithm
                self.classifiers[t] = {'classifier':classifier_cs,
                                       #'weights':curr_weights, 
                                       'eta_t_train':eta_t_train, 
                                       'eta_t_test':eta_t_test 
                                       #'alpha_t':alpha_t
                                       }

            else:
                pass

        ## Return classifier
        return self.classifiers

    def predict_datapoint(self, datapoint, *args, **kwargs):
        '''
        Predict on test data point
        '''
        #print(f'kwargs:{kwargs}'); input()
        pos_label = kwargs.get('pos_label', 'yes')
        neg_label = kwargs.get('neg_label', 'no')
        pos_bin = kwargs.get('pos_bin', 1)
        neg_bin = kwargs.get('neg_bin', -1)
        classifiers = self.classifiers
        final_prediction = 0
        final_prediction_label_hist = {}
        for t, classifier_dic in classifiers.items():
            #print(f'Adding prediction for classifier {t}')
            classifier = classifier_dic['classifier']

            prediction = classifier.predict_datapoint_numeric(datapoint)
            prediction_bin = map_label_to_bin(prediction, pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)
            final_prediction += prediction_bin
            #print(f'final_prediction:{final_prediction}, gt_label:{datapoint[-1]}')
            final_prediction_label = map_bin_to_label(bin_label=final_prediction>0,pos_label=pos_label, neg_label=neg_label, pos_bin=True, neg_bin=False)
            #print(f'final_prediction_label:{final_prediction_label}, gt_label:{datapoint[-1]}')
            final_prediction_label_hist[t] = final_prediction_label



        return final_prediction_label, final_prediction_label_hist

    def predict_dataset(self, dataset, *args, **kwargs):
        '''
        Predict on test dataset
        '''
        pos_label = kwargs.get('pos_label', 'yes')
        neg_label = kwargs.get('neg_label', 'no')
        pos_bin = kwargs.get('pos_bin', 1)
        neg_bin = kwargs.get('neg_bin', -1)
        test_ds = dataset[:]
        predictions = []
        predictions_hist = []
        for i, test_datapoint in enumerate(test_ds):
            #print(f'Predicting datapoint:{i+1}/{len(test_ds)}')
            prediction, prediction_hist = self.predict_datapoint(datapoint=test_datapoint, pos_label=pos_label, neg_label=neg_label, pos_bin=pos_bin, neg_bin=neg_bin)
            test_datapoint = np.hstack((test_datapoint, [prediction]))
            predictions.append(test_datapoint)
            predictions_hist.append(prediction_hist)

        return np.array(predictions), predictions_hist

