"""
Attempting to implement GANs, first for simple 1d probability distributions
and then ideally for MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

# trying out abstract base class stuff
# from collections.abc import abstractmethod

class DataProvider(object):
    """
    A class to provide minibatches of data easily
    """
    def __init__(self, data):
        """
        :attr data (np.ndarray): The data to sample from, as a numpy array
        :attr N (int): The total number of datapoints
        :attr perm (np.ndarray): A permutation of the indices to sample
        :attr count (int): The current index reached when sampling
        """
        self.data = data
        self.N = len(data)
        self.perm = np.random.permutation(self.N)
        self.count = 0

    def reset(self):
        """
        Reset the provider by producing a new permutation
        """
        self.perm = np.random.permutation(self.N)
        self.count = 0

    def sample_mb(self, m):
        """
        Sample a minibatch of data
        
        :param m (int): The number of samples in the minibatch
        :return mb (torch.Tensor): A PyTorch tensor of the sample
        """
        # sample from current permutation
        lo = self.count 
        hi = (self.count + m)
        # handle wrapping
        if hi < self.N:
            indices = self.perm[self.count % self.N: (self.count + m) % self.N]
        else:
            indices = np.concatenate((self.perm[lo % self.N:], self.perm[:hi % self.N]))

        self.count += m
        return torch.from_numpy(self.data[indices])


class GenerativeAdversarialNetwork(object):
    """
    This is a class to house a Generator and a Discriminator and train them jointly.
    I'm still not sure the best way to pass stuff in here but I'll work it out

    """
    def __init__(self, generator, discriminator, data_provider, m, k):
        """
        Constructor for a GAN.
        Takes the two components as input along with training data and some hyperparameters
        
        :attr G (Generator): A generator to be trained
        :attr D (Discriminator): A discriminator to be trained
        :attr data_provider (DataProvider): An object that provides training data minibatches
        :attr m (int): The number of samples to train per minibatch
        :attr k (int): The number of times to train the discriminator per 1 update of the generator
        """
        self.G = generator
        self.D = discriminator
        self.data_provider = data_provider
        self.m = m
        self.k = k

    # DEBUG
    def train_epoch(self, epoch):
        """
        Train for an epoch, returning the average discriminator loss and the only generator loss
        
        :return d_loss (float): Average discriminator loss in the epoch
        :return g_loss (float): Generator loss in the epoch
        """
        # start by resetting the data provider
        self.data_provider.reset()
        # now train the discriminator k times, saving the losses
        d_losses = [None] * self.k
        # DEBUG only training discriminator for a bit
        for i in range(self.k):
            # sample training data
            train_mb = self.data_provider.sample_mb(self.m)
            # sample generator data
            gen_mb = self.G.sample(self.m)
            d_loss = self.D.update(train_mb, gen_mb)
            d_losses[i] = d_loss.item()
        # get average discriminator loss
        d_loss = np.mean(d_losses)
        # train the generator ONCE
        g_loss = self.G.update(self.m, self.D)

        return d_loss, g_loss


class Discriminator(nn.Module):
    """
    This is a general template for discriminators.
    I don't have a lot of experience with this type of programming, but the idea is
    that all discriminators that I use will inherit from this, and this will have abstract methods.
    TODO: figure out if I should restructure this to CONTAIN an nn.Module rather than be one.
    """

    def __init__(self):
        """
        Constructor for Discriminator networks.
        Saving borrowed from RL CW again.

        :attr output_dim (int): Disciminator is always a binary classifier, so this is always 1
        :attr saveables(Dict[str, torch.nn.Module]): stuff to save
        """
        # initialise as a pytorch module
        super(Discriminator, self).__init__()

        self.output_dim = 1
        self.saveables = {}

    def compute_d_loss(self, train_mb, gen_mb):
        """Compute the loss for a minibatch.
        
        :param train_mb (torch.Tensor): minibatch of data from the training distribution, size (batch_size, (data_dims))
        :param gen_mb (torch.Tensor): minibatch of data generated by the generator, size (batch_size, (data_dims))
        :return loss (torch.Tensor): The scalar loss on this minibatch, from which to compute gradients
        """
        # make discriminator predictions for the two minibatches
        train_preds = self(train_mb)
        gen_preds = self(gen_mb)
        # return the loss
        return -torch.mean(torch.log(train_preds) + torch.log(1 - gen_preds))

    # borrowed from RL CW
    def save(self, path):
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models.pt"

        :param path (str): path to directory where to save models
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    # borrowed from RL CW
    def restore(self, save_path):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    # @abstractmethod
    def update(self):
        """
        Every discriminator must have a way to update its parameters
        """
        ...


class MLPDiscriminator(Discriminator):
    """
    This class is for an MLP discriminator (feed-forward neural network).
    It inherits from the Disciminator base class
    """
    def __init__(self, input_dim=1, hidden_dim=10, n_hidden_layers=2, lr=1e-3
                 ):
        # initialise the module
        super(MLPDiscriminator, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # store the layers
        self.hiddens = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.build_module()

        # build an optimiser for this network's params
        self.optim = torch.optim.Adam(params=self.parameters(), lr=lr)

    def build_module(self):
        # simulate a minibatch of size 2 being passed through the network
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

    def update(self, train_mb, gen_mb):
        """
        Update the network's parameters given a minibatch of training examples and generator samples

        :param train_mb (torch.Tensor): minibatch of data from the training distribution, size (batch_size, (data_dims))
        :param gen_mb (torch.Tensor): minibatch of data generated by the generator, size (batch_size, (data_dims))
        :return loss (torch.Tensor): The scalar loss on this minibatch
        """

        # first we zero the gradients, then compute the loss, backpropagate, and take a step 
        self.optim.zero_grad()
        d_loss = self.compute_d_loss(train_mb, gen_mb)
        d_loss.backward()
        self.optim.step()
        return d_loss

class Generator(nn.Module):
    """
    This is a general template for generators.
    I don't have a lot of experience with this type of programming, but the idea is
    that all generators that I use will inherit from this, and this will have abstract methods.
    TODO: figure out if I should restructure this to CONTAIN an nn.Module rather than be one.
    """
    def __init__(self, noise_prior, input_dim, output_dim):
        """
        Constructor for Generator networks.
        Saving borrowed from RL CW again.

        :attr input_dim: the size of the noise to pass to the generator
        :attr output_dim: the size to output, i.e. the dimensions of the training data
        :attr noise_prior (torch.distributions): An (instantiated) torch distribution to sample noise from
        :attr saveables (Dict[str, torch.nn.Module]): stuff to save
        """
        # initialise as a pytorch module
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_prior = noise_prior

        self.saveables = {}

    def compute_g_loss(self, gen_mb, discriminator):
        """Compute the loss for a minibatch.
        
        :param gen_mb (torch.Tensor): minibatch of data generated by the generator, size (batch_size, (data_dims))
        :param Discriminator (Discriminator): a Discriminator subclass to make predictions
        :return loss (torch.Tensor): The scalar loss on this minibatch, from which to compute gradients
        """
        # make discriminator predictions for the two minibatches
        gen_preds = discriminator(gen_mb)
        # NOTE: testing filtering out nan inf values
        raw_logs = torch.log(gen_preds)
        safe_logs = torch.where(torch.isnan(raw_logs) | (raw_logs == float('-inf')), torch.zeros_like(raw_logs), raw_logs)
        return -torch.mean(safe_logs)

    # borrowed from RL CW
    def save(self, path):
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models.pt"

        :param path (str): path to directory where to save models
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    # borrowed from RL CW
    def restore(self, save_path):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    def sample(self, m, training=True):
        """
        Sample a minibatch of outputs from the generator

        :param m (int): the number of samples to draw
        """
        # set the mode correctly
        if training:
            self.train()
        else:
            self.eval()
        # produce noise from the noise prior
        # NOTE TODO: for now this only accepts 1d input dim
        z_mb = self.noise_prior.sample(sample_shape=(m, self.input_dim))
        # produce output by calling the generator
        gen_mb = self(z_mb)

        return gen_mb

    # @abstractmethod
    def update(self):
        """
        Every generator must have a way to update its parameters
        """
        ...

    

class MLPGenerator(Generator):
    """
    This class is for an MLP Generator (i.e. feedforward neural network)
    Its output size is the same as the training data, and its input size is (probably) that size too.
    """

    def __init__(self, noise_prior, input_dim=1, output_dim=1, hidden_dim=10, n_hidden_layers=2, lr=1e-3
                 ):
        # initialise the module
        super(MLPGenerator, self).__init__(noise_prior, input_dim, output_dim)

        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

        # store the layers
        self.hiddens = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.build_module()

        self.optim = torch.optim.Adam(params=self.parameters(), lr=lr)

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
        # apply sigmoid to get each pixel between 0 and 1
        out = torch.sigmoid(out)
        return out

    def update(self, m, discriminator):
        """
        Update the generator based on the current discriminator 

        :param m (int): the size of the minibatch to train on
        :param discriminator (Discriminator): A discriminator subclass to make predictions
        :return loss (torch.Tensor): The generator loss from this training step
        """
        self.optim.zero_grad()
        gen_mb = self.sample(m)
        g_loss = self.compute_g_loss(gen_mb, discriminator)
        g_loss.backward()
        self.optim.step()
        return g_loss
