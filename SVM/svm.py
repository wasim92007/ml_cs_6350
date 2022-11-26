## External library import
import os, sys
import numpy as np
import random
import copy
import scipy
import scipy.optimize
from math import exp

np.random.seed(1234)

class Primal_SVM(object):
    '''
    Class for SVM in primal domain with stocastic
    sub-gradient descent
    '''
    def __init__(self, lr_scheduler, c, max_epochs=5) -> None:
        '''
        Initialize primal svm
        '''
        self.lr_scheduler = lr_scheduler
        self.c            = c
        self.max_epochs   = max_epochs
        self.w            = None
        self.lr_init      = None
        self.curr_lr      = None

    def train(self, train_ds, *args, **kwargs):
        '''
        Train the primal svm
        '''

        ## Get the number of features
        self.num_feats = len(train_ds[0]) - 1

        ## Initialize the weight
        if self.w is None:
            #self.w = kwargs.get('init_w', np.random.rand(1, self.num_feats+1))
            self.w = kwargs.get('init_w', np.zeros((1, self.num_feats+1)))

        ## Initialize learning rate
        if self.lr_init is None:
            self.lr_init = kwargs.get('lr_init', 1)
            self.curr_lr = self.lr_init

        ## Number of taining dataset
        self.num_train_ds  = len(train_ds)
        self.random_idexes = [i for i in range(self.num_train_ds)]

        ## Train for max epochs
        for epoch in range(self.max_epochs):
            #if epoch % 20 == 0 or epoch == self.max_epochs-1:
            #    print(f'Training epoch {epoch+1}/{self.max_epochs}')

            ## Get current learning rate
            self.curr_lr = self.lr_scheduler(epoch)

            ## Random shuffle the datapoints
            np.random.shuffle(self.random_idexes)

            for idx in self.random_idexes:

                ## Get the feats, y
                feats = train_ds[idx][:-1]
                y     = train_ds[idx][-1]

                ## Extend the feats with 1
                x = np.hstack((feats, [1]))[np.newaxis, :]

                ## Perform dot product
                wx = np.dot(x, self.w.transpose()) 

                if y * wx[0][0] <= 1:
                    ## Get current weight
                    curr_w = copy.deepcopy(self.w)

                    ## Make the bias term 0
                    curr_w[:,-1] = 0
                    
                    ## Update weight
                    self.w = self.w - self.curr_lr * curr_w + self.curr_lr * self.num_train_ds * self.c * y * x
                    #self.w = self.w - self.curr_lr * curr_w + self.curr_lr * self.c * y * x

                else:
                    self.w = (1-self.curr_lr) * self.w

    def test(self, test_ds, *args, **kwargs):
        '''
        Test the primal svm
        '''
        X = copy.deepcopy(test_ds) 
        ## Extend feature with 1
        X[:, -1] = 1

        preds = []
        for x in X:
            ## Make prediction on datapoint
            pred = np.sign(np.dot(x, self.w.transpose()))

            ## Add to prediction list
            preds.append(pred[0])

        return np.array(preds)

def get_gaussian(x, y, gamma):
    return exp(-(np.linalg.norm(x-y, ord=2)**2) / gamma)

class Dual_SVM(object):
    '''
    Class for SVM in dual domain with constraint optimization
    '''
    def __init__(self) -> None:
        '''
        Initialize the dual svm
        '''
        self.w_star          = None
        self.b_star          = None
        self.support_vectors = []

    def get_gaussian_kernel(self, x, y , gamma):
        '''
        Get Gaussian kernel transform
        '''
        #gk = np.linalg.norm(x - y, ord=2)
        #gk = - gk**2
        #gk /= gamma
        #gk = math.exp(gk)
        gk = exp(-(np.linalg.norm(x - y, ord=2) ** 2) / gamma)        
        #gk = exp(- np.dot(x-y, x-y)/gamma)

        return gk

    def train(self, train_ds, c, *args, tolerance=1e-10, **kwargs):
        '''
        Train the dual svm
        '''
        kernel = kwargs.get('kernel', 'dot_prod')
        gamma  = kwargs.get('gamma', None)

        feats = copy.deepcopy(train_ds[:,:-1])
        y     = copy.deepcopy(train_ds[:,-1])

        def inner_optimization_func(a, feats, y):
            '''
            Innner optimization function: alpha, x, y
            '''
            ## Get the length of a and y
            len_a = len(a)
            len_y = len(y)

            ## Define the matrixes
            matrix_a = np.ones((len_a, len_a)) * a
            matrix_y = np.ones((len_y, len_y)) * y

            ## Perform kernel trick
            if kernel == 'gaussian':
                #print(f'gamma:{gamma}')
                feats_out_prod = feats**2 @ np.ones_like(feats.T) - 2 * feats@feats.T + np.ones_like(feats) @ feats.T**2  ## (feat - feat.T)^2
                feats_out_prod = np.exp(-(feats_out_prod/gamma))
            else:
                feats_out_prod = feats@feats.T  ## Identity mapping

            obj_val = (matrix_y * matrix_y.T) * (matrix_a * matrix_a.T) * feats_out_prod
            obj_val = 0.5 * np.sum(obj_val) - np.sum(a)

            return obj_val

        optimization_constraints = [{'type':'ineq', 'fun': lambda a : a},     ## alpha > 0
                                    {'type':'ineq', 'fun': lambda a : c - a}, ## alpha < c
                                    {'type':'eq',   'fun': lambda a : np.dot(a, y)}  ## sum_i(alpha_i*y_i) = 0
                                    #{'type':'eq',   'fun': lambda a : np.sum(a*y)}  ## sum_i(alpha_i*y_i) = 0
                                   ]

        ## Perform minimization of the inner optimization function and find
        ## Lagrange multiplier alpha*
        opt_res = scipy.optimize.minimize(inner_optimization_func, x0=np.zeros((len(feats),)), args=(feats, y), method='SLSQP', constraints=optimization_constraints, tol=0.001)

        ## Get the optimial weight w* = sum_i(alpha_start_i * y_i * x_i)
        self.w_star = np.sum([opt_res['x'][i] * y[i] * feats[i] for i in range(len(feats))], axis=0)
        print(f'w*:{self.w_star}')

        ## Get the optimial bias b* = y_j - w_start.T * x_j
        if kernel == 'gaussian':
            self.b_star = np.mean([y[i] - self.get_gaussian_kernel(x=self.w_star, y=feats[i], gamma=gamma) for i in range(len(feats))])
        else:
            self.b_star = np.mean([y[i] - np.dot(self.w_star, feats[i]) for i in range(len(feats))])
        print(f'b*:{self.b_star}')

        for i, a in enumerate(opt_res['x']):
            #print(f'a:{a}')
            #input()
            if a > tolerance:
                self.support_vectors.append(feats[i])

    def test(self, test_ds, *args, **kwargs):
        '''
        Test the dual svm
        '''
        kernel = kwargs.get('kernel', 'dot_prod')
        gamma  = kwargs.get('gamma', None)

        feats = copy.deepcopy(test_ds[:,:-1])
        predictions = []
        print(f'kernel:{kernel}, gamma:{gamma}, w*:{self.w_star}, b*{self.b_star}')
        for datapoint in feats:
            if kernel == 'gaussian':
                gk = self.get_gaussian_kernel(x=self.w_star, y=datapoint, gamma=gamma)
                #gk = get_gaussian(x=self.w_star, y=datapoint, gamma=gamma)
                prediction = np.sign(gk + self.b_star)
                #print(f'datapoint:{datapoint}')
                #print(f'kernel:{kernel}, gamma:{gamma}, w*:{self.w_star}, b*{self.b_star}, gk:{gk}, prediction:{prediction}')
                predictions.append(prediction)
                #input()
            else:
                predictions.append(np.sign(np.dot(self.w_star, datapoint) + self.b_star))

        return np.array(predictions)

        