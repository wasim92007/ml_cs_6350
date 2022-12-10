## Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class BankNote_DS(Dataset):
    '''
    Pytorch bank-note dataset
    '''
    def __init__(self, fname):
        ## Parse the file into feats and labels
        feats, labels = [], []
        with open(fname, "r") as f:
            for line in f:
                fields = line.strip().split(",")
                fields_float = list(map(lambda x : np.float32(x), fields))
                feats.append(fields_float[:-1])
                labels.append(fields_float[-1])

        self.feats = np.array(feats)
        self.labels = np.array(labels)
    
    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat = self.feats[idx]
        label = self.labels[idx]
        return feat, label


def he_init(layer):
    '''
    He initialization
    '''
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.fill_(0.01)


def xavier_init(layer):
    '''
    
    '''
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)