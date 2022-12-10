## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *
import homemade_nn as nn

if __name__ == '__main__':
    print(f'Computing gradient using backpropagation for paper problem 3')

    model = nn.NN_Sequential([
        nn.FC_Layer(in_channels=2, out_channels=2, act_func='sigmoid', initialization='normal'),
        nn.FC_Layer(in_channels=2, out_channels=2, act_func='sigmoid', initialization='normal'),
        nn.FC_Layer(in_channels=2, out_channels=2, act_func='sigmoid', initialization='normal'),
        nn.FC_Layer(in_channels=2, out_channels=1, act_func='linear', initialization='normal', include_bias=False)
    ])

    #x = np.array([1,1,1])
    x = np.array([1,1]) ## Convention x0 = 1
    ystar = 1

    ## Forward pass
    y, z = model.forward(x)

    ## Backward pass
    model.backward(z=z, y=ystar, lr=1, print_grad=True)



    print(f'Computing gradient using backpropagation for bank-note 3')

    model = nn.NN_Sequential([
        nn.FC_Layer(in_channels=4, out_channels=4, act_func='sigmoid', initialization='normal'),
        nn.FC_Layer(in_channels=4, out_channels=4, act_func='sigmoid', initialization='normal'),
        nn.FC_Layer(in_channels=4, out_channels=4, act_func='sigmoid', initialization='normal'),
        nn.FC_Layer(in_channels=4, out_channels=1, act_func='linear', initialization='normal', include_bias=False)
    ])

    #x = np.array([1,1,1])
    ## Firsttraining example
    x = np.array([3.8481,10.1539,-3.8561,-4.2228]) ## Convention x0 = 1
    ystar = 0

    ## Forward pass
    y, z = model.forward(x)

    ## Backward pass
    model.backward(z=z, y=ystar, lr=1, print_grad=True)


