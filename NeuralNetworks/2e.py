## Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common_functions import *
from dataprocessing import *


def train_epoch(model, dataloader, loss_fn, optimizer):
    ## Put model to training mode
    model.train()

    train_loss = []
    for batch, (feats, labels) in enumerate(dataloader):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)

        ## Perform model prediction
        pred_labels = model(feats)

        ## Calculate prediction loss
        pred_labels = torch.reshape(pred_labels, labels.shape)  ## Reshape
        loss = loss_fn(pred_labels, labels)

        ## Reset optimizer grad
        optimizer.zero_grad()

        ## Back propagation
        loss.backward()

        ## Update weights
        optimizer.step()

        ## Record every 10 batches
        if batch % 10 == 0:
            train_loss.append(loss.item())

    #print(f'Training error: {np.mean(train_loss):>6f}')

    return train_loss

def test(model, dataloader, loss_fn):
    ## Put model to evaluation mode
    model.eval()

    test_loss = 0

    with torch.no_grad():   ## Disable gradient calculation
        for feats, labels in dataloader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            ## Perform model prediction
            pred_labels = model(feats)
            pred_labels = torch.reshape(pred_labels, labels.shape)

            ## Accumulate loss for entire dataset
            test_loss += loss_fn(pred_labels, labels).item()

    num_batches = len(dataloader)
    test_loss /= num_batches

    #print(f'Test error: {test_loss:>6f} \n')
    return test_loss

if __name__ == '__main__':
    ## Hyper-parameters
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    BATCH_SIZE = 8

    ## List of activation function
    activation_func_list = [(nn.ReLU(), he_init, 'ReLU'), (nn.Tanh(), xavier_init, 'Tanh')]

    ## Get the datasets
    train_dataset = BankNote_DS('./data/bank-note/train.csv')
    test_dataset  = BankNote_DS('./data/bank-note/test.csv')

    ## Get the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    ## Get GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {DEVICE} as device')

    ## Try different activation function
    for act_func, init_func, act_name in activation_func_list:
        print(f'Using activation function: {act_name}')

        ## Try different width
        for width in widths:

            ## Try different depth
            for depth in depths:

                print(f'Using NN of depth:{depth} and width:{width}')

                ## Define the NN 
                class BankNote_Classifier(nn.Module):
                    '''
                    Bank-Note Classifier Model
                    '''
                    def __init__(self) -> None:
                        super(BankNote_Classifier, self).__init__()

                        ## Define layers
                        #self.input_layer = nn.Sequential(nn.Linear(4, width), act_func) ## Input layer
                        self.input_layer = nn.Sequential(nn.Linear(4, width)) ## Input layer: Following class
                        self.deep_layers = nn.ModuleList()  ## Placeholder for deep layers

                        ## Add the deep layers
                        for i in range(depth - 2): ## Exclusing input and output layers
                            self.deep_layers.append(nn.Sequential(nn.Linear(width, width), act_func))

                        ## Add output layer
                        self.output_layer = nn.Linear(width, 1)

                
                    ## Define forward pass function
                    def forward(self, feat):
                        ## Pass through input layer
                        out = self.input_layer(feat)
                        ## Pass through hidden layers
                        for hidden_layer in self.deep_layers:
                            out = hidden_layer(out)

                        ## Pass through Output layer
                        out = self.output_layer(out)

                        ## Return output
                        return out

                ## Instantiate the model
                model = BankNote_Classifier().to(DEVICE)
                
                ## Change initialization
                model.apply(init_func)

                ## Define loss function
                loss_fn = nn.MSELoss()

                ## Define optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                train_losses = np.array([])

                ## Train for epochs
                for epoch in range(EPOCHS):

                    ## Train one epoch
                    epoch_loss = train_epoch(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
                    #print(f'epoch_loss:{epoch_loss}')
                    #print(f'Epoch:{epoch}, Training error: {np.mean(epoch_loss):>8f}')
                    #print(f'Training epoch:{epoch}, training loss:{np.mean(epoch_loss):>6f}')

                    ## Append losses
                    train_losses = np.append(train_losses, epoch_loss)

                    ## Create result directory if it does not exist
                    create_dir('results')

                ## Plot graphs
                fig, ax = plt.subplots()
                ax.plot(train_losses)
                ax.set_title(f'PyTorch NN depth:{depth}, width:{width}')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('MSE Loss')
                plt.savefig(f'./results/torch_nn_act_func_{act_name}_depth_{depth}_width_{width}.png')
                plt.close()
        
                train_loss = test(model=model, dataloader=train_dataloader, loss_fn=loss_fn)
                test_loss = test(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
                print(f'PyTorch NN with {act_name} activation, depth:{depth}, width:{width}\nTraining Loss:{train_loss:>6f}, Testing loss:{test_loss:>6f}')
