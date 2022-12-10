## External library import
import os, sys
import numpy as np

## Internal library import
from common_functions import *

## The idea for this implementation is that we will mimic gradient calculation
## explicitly

## As we do not want to use pytorch functions
## We we create thos functions from scratch, so style and
## variable naming convension we will try to keep same

## Please not we need to use float128 to handle overflow

## I have took some inspiration from the online tutorial --
## https://www.pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python


## Let us implement the two most common activation function
class sigmoid_act:
    '''
    Sigmoid activation function
    '''

    def __call__(self, x):
        '''
        Apply sigmoid activation
        '''
        z   = np.exp(-x)    ## Take exponent of -x
        sig = 1 / (1+z)     ## Sigmoid
        return sig

    def grad(self, x):
        '''
        Return gradient
        '''
        z   = np.exp(-x)    ## Take exponent of -x
        sig = 1 / (1+z)     ## Sigmoid

        return sig * (1-sig)

class linear_act:
    '''
    Linear activation function
    '''

    def __call__(self, x):
        '''
        Apply linear activation function
        '''
        return x

    def grad(self, x):
        '''
        Return gradient
        '''
        return 1

## Now that we have out activation function with gradient
## Let us have a fully connected layer with gradient, similar to pytorch
## but with option for activation fucntion and weight initialization
class FC_Layer:
    '''
    Fully connected layer
    '''
    def __init__(self, in_channels, out_channels, include_bias=True, act_func='linear', initialization='zeros'):
        '''
        Initialize the fully connected layer
        '''

        ## Passed parameters
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.include_bias   = include_bias
        self.act_func       = act_func
        self.initialization = initialization

        ## Derived parameters
        if self.include_bias:
            self.layer_shape = (self.in_channels+1, self.out_channels+1)
        else:
            self.layer_shape = (self.in_channels+1, self.out_channels)

        ## Weights
        if self.initialization == 'normal':
            self.weights = np.random.standard_normal(self.layer_shape).astype(np.float128)
        elif self.initialization == 'zeros':
            self.weights = np.zeros(self.layer_shape, dtype=np.float128)
        else:
            raise NotImplementedError

        ## Activation function
        if self.act_func == 'linear':
            self.activation = linear_act()
        elif self.act_func == 'sigmoid':
            self.activation = sigmoid_act()

    def __str__(self)-> str:
        '''
        When printed prints the layer weights
        '''
        return str(self.weights)

    def forward(self, x):
        '''
        Forward pass
        '''
        dp = np.dot(x, self.weights)
        out = self.activation(dp)
        return out

    def backward(self, z, partial_derivs):
        '''
        Backward pass
        '''
        grad = np.dot(partial_derivs[-1], self.weights.T)
        grad = grad * self.activation.grad(z)
        return grad

    def update_weights(self, z, lr, partial_derivs):
        '''
        Update weights
        '''
        grad = np.dot(z.T, partial_derivs)
        self.weights += -1 * lr * grad

        return grad

## Now that we have a custom fully connected layer,
## let us try to build a Neural network using it
## This will mimic the pytorch squential layer
class NN_Sequential():
    '''
    Dense multi-layer perceptron neural network using 
    backpropagatin via nueral network
    Takes layers as input, following pytorch convension
    '''

    def __init__(self, layers):
        '''
        Get the layers to be wrapped inside sequential wrapper
        '''
        self.layers = layers

    def forward(self, x):
        ## Add bias
        _x = np.append(1, x)
        #print(f'_x:{_x}, shape:{_x.shape}')

        ## Reshape x
        z = [np.atleast_2d(_x)]

        ## Pass through the layers sequentially
        for i in range(len(self.layers)):
            #print(f'Forward pass layer:{i}, shape:{self.layers[i].layer_shape}, weights shape:{self.layers[i].weights.shape}')
            out = self.layers[i].forward(z[i])
            z.append(out)

        return z[-1].astype(np.float32), z

    def backward(self, z, y, lr=0.01, print_grad=False):
        '''
        Backward pass
        '''
        partial_derivs = [z[-1] - y] ## At output node: y - y*

        ## Calculate partial derivative from output to input layers
        for i in range(len(z)-2, 0, -1): ## Excluding input, output layer
            grad = self.layers[i].backward(z[i], partial_derivs)
            partial_derivs.append(grad)

        ## Reverse the direction
        partial_derivs = partial_derivs[::-1]

        ## Calculate gradient
        for i in range(len(self.layers)):
            grad = self.layers[i].update_weights(z=z[i], lr=lr, partial_derivs=partial_derivs[i])
            if print_grad:
                print(f'Gradient of layer {i+1}:\n{grad}')






