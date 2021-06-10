
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #1st Layer
        out = self.l1(x)
        out = self.relu(out)
        #2nd Layer
        out = self.l2(out)
        out = self.relu(out)
        #3rd Layer
        out = self.l3(out)

        return out     

    
