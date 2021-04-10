"""
Attempting to implement GANs, first for simple 1d probability distributions
and then ideally for MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorNetwork(nn.Module):
    """
    This class is for the discriminator. 
    This is a binary classifier: did this input come from the training data or not?
    """
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1, n_hidden_layers=2, lr=1e-3
                 ):
        # initialise the module
        super(DiscriminatorNetwork, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # store the layers
        self.hiddens = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.build_module()

    def build_module(self):
        # simulate a minibatch being passed through the network
        x = torch.zeros((2, self.input_dim))
        out = x
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        out = self.input_layer(out)
        # we subtract one because the last hidden layer goes to the output
        for i in range(self.n_hidden_layers - 1):
            # calculate the dimension of the batchnorm needed
            batch_dim = out.size(-1)
            self.batchnorms.append(nn.BatchNorm1d(batch_dim))
            out = self.batchnorms[i](out)
            self.hiddens.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            out = self.hiddens[i](out)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, X):
        out = X
        out = self.input_layer(out)
        out = F.relu(out)
        for i in range(self.n_hidden_layers - 1):
            out = self.batchnorms[i](out)
            out = self.hiddens[i](out)
            out = F.relu(out)
        # on the last output, we apply a sigmoid to get a binary prediction
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        return out

class GeneratorNetwork(nn.Module):
    """
    This class if for the discriminator.
    Its output size is the same as the training data, and its input size is (probably) that size too.
    """

    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1, n_hidden_layers=2, lr=1e-3
                 ):
        # initialise the module
        super(GeneratorNetwork, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # store the layers
        self.hiddens = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.build_module()

    def build_module(self):
        # simulate a minibatch being passed through the network
        x = torch.zeros((2, self.input_dim))
        out = x
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        out = self.input_layer(out)
        # we subtract one because the last hidden layer goes to the output
        for i in range(self.n_hidden_layers - 1):
            # calculate the dimension of the batchnorm needed
            batch_dim = out.size(-1)
            self.batchnorms.append(nn.BatchNorm1d(batch_dim))
            out = self.batchnorms[i](out)
            self.hiddens.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            out = self.hiddens[i](out)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, X):
        out = X
        out = self.input_layer(out)
        out = F.relu(out)
        for i in range(self.n_hidden_layers - 1):
            out = self.batchnorms[i](out)
            out = self.hiddens[i](out)
            out = F.relu(out)
        out = self.output_layer(out)
        return out
