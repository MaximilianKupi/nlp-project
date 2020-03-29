import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.linear import Linear
from torch.optim.adam import Adam
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot, make_dot_from_trace
import json

class CNN_1d_experiment(nn.Module):
    # Convolutional neural network (two convolutional layers)
    def __init__(self,variables):
        """ Initializes the CNN as a class object
        """        
        super(CNN_1d_experiment, self).__init__()
        self.layer1 = nn.Sequential( # first layer of CNN
            nn.Conv1d(**{
                    "in_channels" : 120,
                    "out_channels" : 16,
                    "kernel_size" : 1
                }), 
            nn.BatchNorm1d(**{
                    "num_features" : 16
                }),
            nn.ReLU(), # activation function
            #nn.MaxPool1d(**layer1_arguments["MaxPool2d"])
        )
        print(self.layer1)

        self.layer2 = nn.Sequential( # second layer of CNN
            nn.Conv1d(**{
                    "in_channels" : 16,
                    "out_channels" : 32,
                    "kernel_size" : 1,
                }), 
            nn.BatchNorm1d(**{
                    "num_features" : 32
                }),
            nn.ReLU(),
            # nn.MaxPool1d(**layer2_arguments["MaxPool2d"])
        )
        print(self.layer2)

        self.fc = nn.Linear(**{
            "in_features" : 32,
            "out_features" : 3
        }) # final layer of CNN
        print(self.fc)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out