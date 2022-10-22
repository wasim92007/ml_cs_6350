import math
import random
import numpy as np
from numpy import linalg as LA

from visualization import *

class Dataset(object):
    '''
    Dataset Class
    '''
    def __init__(self, dataset, *args, **kwargs) -> None:
        self.dataset    = dataset
        self.n_dp       = len(dataset)
        self.batch_size = kwargs.get('batch_size', self.n_dp)
        self.idx        = 0
        self.use_sgd    = kwargs.get('use_sgd', False)
        if self.use_sgd:
            print(f'Using stochastic gradient decent(SGD)')

    def __next__(self):
        feats, labels = [], []
        if self.idx >= self.n_dp:
            self.idx = 0
            raise StopIteration
        for i in range(self.batch_size):
            try:
                if self.use_sgd:
                    idx = random.randint(0,self.n_dp-1)
                else:
                    idx = self.idx
                feats.append (self.dataset[idx][:-1])
                labels.append(self.dataset[idx][-1])
                self.idx += 1
            except:
                break
        
        feats, labels = np.array(feats), np.array(labels)[:,np.newaxis]
        feats = np.hstack((feats, np.ones((len(feats),1))))

        return feats, labels

    def __iter__(self):
        return self

    def __len__(self):
        return math.ceil(self.n_dp/self.batch_size)


class LMS_Linear_Regression(object):
    '''
    Class for LMS linear regression
    '''
    def __init__(self, lr=0.001, n_epoch=200, batch_size=4, *args, **kwargs) -> None:
        '''
        Initialize LMS linear regression class
        '''
        self.lr         = lr
        self.n_epoch    = n_epoch
        self.batch_size = batch_size
        self.w          = None

        self.args       = args
        self.kwargs     = kwargs

    def initialize_w(self, w):
        '''
        Initialize weight
        '''
        self.w = np.array(w)

    def get_w(self):
        '''
        Returns the weight
        '''
        return self.w

    def get_loss(self, pred, gt, *args, **kwargs):
        '''
        Calculate the loss
        '''
        #print(f'gt: {gt}')
        #print(f'pred: {pred}')
        
        ## Get the difference
        diff = gt - pred
        #print(f'diff: {diff}')
        
        ## Square the difference
        sqr_diff = np.square(diff)
        #print(f'sqr_diff: {sqr_diff}')
        #print(f'diff[0]**2:{diff[0]**2}')

        ## Take the sum of squares
        loss = np.sum(sqr_diff)

        ## Multiply with 0.5
        loss = 0.5 * loss

        #print(f'loss:{loss}')
        return loss

    def get_grad(self, x, y, w):
        '''
        Calculates the gradient
        '''
        #print(f'x:{x}')
        wx = np.dot(x, w.transpose())
        diff = y - wx
        #print(f'diff:{diff}')
        #print(f'diff shape:{diff.shape}')
        grad = -1 * diff * x
        #print(f'grad:{grad}')
        #print(f'grad shape:{grad.shape}')

        grad = np.sum(grad, axis=0)

        return grad

    def train(self, train_ds, *args, **kwargs):
        '''
        Train using LMS linear regression
        '''
        #### Get hyper-parameters
        ## Convergence parameters
        till_convergence = kwargs.get('till_convergence', True)
        if till_convergence:
            self.n_epoch = 10000
        tolerence = kwargs.get('tolerance', 10e-6)

        ## SDG parameters
        use_sgd = kwargs.get('use_sgd', False)

        ## Weight initialization
        n_feat = len(train_ds[0])
        if self.w is None:
            self.w = kwargs.get('init_w', np.random.rand(1, n_feat))
        #self.w = kwargs.get('init_w', np.random.rand(1, n_feat))
        #self.w = kwargs.get('init_w', np.ones((1, n_feat)))
        #print(f'Initial weight: {self.w}')
        #print(f'init_w shape:{self.w.shape}')

        epoch = 0
        train_ds_iter = Dataset(dataset=train_ds, batch_size=self.batch_size, use_sgd=use_sgd)
        n_batch     = len(train_ds_iter)
        epoch_loss_list = []
        tolerence_met = False
        w_new_list = []
        grad_list  = []
        converged = False
        while not converged:
            batch_idx = 0
            batch_loss_list = []
            for feats, labels in train_ds_iter:

                ## Batch Processing: Begin
                #print(f'feats:{feats}')
                #print(f'labels:{labels}')
                #print(f'feats shape:{feats.shape}')
                #print(f'labels shape:{labels.shape}')

                ## Calculate x*transpose(w)
                #print(f'w:{self.w}')
                wx = np.dot(feats, self.w.transpose())
                #print(f'wx:{wx}')
                #print(f'wx shape:{wx.shape}')

                ## Calculate loss
                batch_loss = self.get_loss(pred=wx, gt=labels)
                batch_loss_list.append(batch_loss)
                #print(f'batch loss:{batch_loss}')
                #print(f'Batch number: {batch_idx+1}/{n_batch}, loss:{batch_loss}')

                ## Calculate grad
                grad = self.get_grad(x=feats, y=labels, w=self.w)
                #print(f'grad:{grad}')
                grad_list.append(grad)

                w_new = self.w - self.lr * grad
                #print(f'w_new:{w_new}')
                w_new_list.append(w_new)
                w_norm = LA.norm(self.w - w_new)
                #print(f'w_norm:{w_norm}')
                if till_convergence and w_norm <= tolerence:
                    print(f'Torelence met:{tolerence}')
                    tolerence_met = True
                
                ## Calculate updated weights
                self.w = w_new

                ## Batch Processing: End
                batch_idx += 1
            epoch_loss = np.average(batch_loss_list)
            print(f'Training epoch: {epoch}, Loss:{epoch_loss}')
            epoch_loss_list.append(epoch_loss)
            
            epoch += 1
            if not tolerence_met:
                converged = epoch == self.n_epoch
            else:
                converged = True

        graph_name = kwargs.get('graph_name', 'loss.png')
        label_dic  = kwargs.get('label_dic', {'xlabel': 'Iteration(t)', 'ylabel': 'Loss', 'legend':['train'], 'title':f'LMS Linear Reg Loss (lr={self.lr}), SGD:{use_sgd}'})

        plot_one_data_func(data_1=epoch_loss_list, graph_name=graph_name, label_dic=label_dic) 
        #print(f'Grad_list:{grad_list}')
        #print(f'w_new_list:{w_new_list}')

        return epoch_loss_list

    def test_datapoint(self, test_dp, *args, **kwargs):
        '''
        Test datapoint using LMS linear regression 
        '''
        pass

    def test_dataset(self, test_ds, *args, **kwargs):
        '''
        Test dataset using LMS linear regression 
        '''
        predictions = []
        losses      = []

        test_ds_iter = Dataset(dataset=test_ds, batch_size=len(test_ds))
        for feats, labels in test_ds_iter:

            ## Batch Processing: Begin
            wx = np.dot(feats, self.w.transpose())

            ## Calculate loss
            batch_loss = self.get_loss(pred=wx, gt=labels)

            ## Add prediction to result
            predictions.append(np.hstack((feats, wx)))
            losses.append(batch_loss)
            ## Batch Processing: End

        return np.array(predictions), np.array(losses)