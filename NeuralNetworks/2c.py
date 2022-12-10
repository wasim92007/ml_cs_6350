## External library import
import os, sys
import numpy as np
import matplotlib.pyplot as plt


## Internal library import
from common_functions import *
import homemade_nn as nn

def train(train_x, train_y, model, n_epochs, lr_0 = 0.5, d = 1):
    epoch_losses = []

    ## Get all the indexes
    idxs = np.arange(len(train_x))
    for epoch in range(n_epochs):
        per_epoch_losses = []

        ## Shuffle the indexes
        np.random.shuffle(idxs)
        
        ## Interate through every training example
        for i in idxs:

            ## Get model prediction
            y, z = model.forward(train_x[i])

            ## Append loss 
            loss = get_square_loss(preds=y, labels=train_y[i])
            per_epoch_losses.append(loss)

            ## Get the leraning rate from schedule
            lr = lr_0 / (1 + (lr_0 / d) * epoch)

            ## Backpropagation and weights update
            model.backward(z, train_y[i], lr)

        ## Append average loss of every epoch
        epoch_losses.append(np.mean(per_epoch_losses))
        #print(f"Epoch {epoch+1} training error: {np.mean(per_epoch_losses):>6f}")
    
    return epoch_losses

def test(test_x, test_y, model):
    test_losses = []

    ## Interate through every testing example
    for i in range(len(test_x)):

        ## Get model prediction
        y, _ = model.forward(test_x[i])

        #print(f'test_y[i]:{test_y[i]}, y:{y}, {(y>0.5).astype(np.float32)}')
        #input()
        ## Append loss 
        #loss = get_square_loss(y, test_y[i])
        loss = ((y>0.5).astype(np.float32) == test_y[i]).astype(np.float32)
        test_losses.append(loss)

    #print(f"Testing error: {np.mean(test_losses):>6f}\n")

    return 1 - np.mean(test_losses)

if __name__ == '__main__':
    ## Hyper-parameters
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    LEARNING_RATE = 1e-3
    EPOCHS = 40
    BATCH_SIZE = 8

    train_dataset = open_csv_file('./data/bank-note/train.csv', numeric_val=True)
    test_dataset  = open_csv_file('./data/bank-note/test.csv', numeric_val=True)

    #print(f'train_dataset:{train_dataset[:2]}, shape:{train_dataset.shape}, dtype:{train_dataset.dtype}')

    train_x, train_y = train_dataset[:,:-1], train_dataset[:,-1]
    test_x, test_y   = test_dataset[:,:-1],  test_dataset[:,-1]

    for width in widths:
        print(f'Network width:{width}')

        ## Get the model
        model = nn.NN_Sequential([
            nn.FC_Layer(in_channels=4, out_channels=width, act_func='sigmoid', initialization='zeros'),
            nn.FC_Layer(in_channels=width, out_channels=width, act_func='sigmoid', initialization='zeros'),
            nn.FC_Layer(in_channels=width, out_channels=width, act_func='sigmoid', initialization='zeros'),
            nn.FC_Layer(in_channels=width, out_channels=1, act_func='linear', initialization='zeros', include_bias=False),
        ])

        ## Train model
        train_losses = train(train_x, train_y, model, n_epochs=EPOCHS, lr_0 = 0.5, d = 1)

        test_losses  = test(test_x, test_y, model)

        ## Plot graphs
        fig, ax = plt.subplots()
        ax.plot(train_losses)
        ax.set_title(f'Homemade NN depth:2, width:{width}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE Loss')
        plt.savefig(f'./results/homemade_nn_act_func_sigmoid_zero_init_depth_2_width_{width}.png')
        plt.close()
        
        train_loss = test(train_x, train_y, model)
        test_loss  = test(test_x, test_y, model)
        print(f'Homemade NN with sigmoid activation, depth:2, width:{width}\nTraining Error:{train_loss:>6f}, Testing Errro:{test_loss:>6f}')
