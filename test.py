"""
Testing what I've written for the discriminator and generator
"""

from gan import DiscriminatorNetwork, GeneratorNetwork
import torch
import matplotlib.pyplot as plt
import numpy as np


# points to evaluate the network at
n_eval = 1000
x_eval = torch.linspace(0, 1, n_eval).reshape(-1, 1)

# # Visualising samples from the prior implicitly made by random weights
# N = 10
# outputs = []
# for i in range(N):
#     D = DiscriminatorNetwork(hidden_dim=50)
#     outputs.append(D(eval_x))
#     plt.plot(eval_x, outputs[i].detach())
# plt.show()

# training data is 1d samples from a unit gaussian (with bias away from 0)

# Visualising the distribution of the training data
# plt.hist(np.array(training_x.reshape(-1)), bins=30)
# plt.show()

n_train = 2000
mean, var = 0.5, 0.1
x_train = torch.normal(mean, var, (n_train, 1))
# for now, just always saying x = 0.2
# x_train = torch.full(size=(n_train, 1), fill_value=0.22)

# Training the discriminator against a very poor generator (just a uniform distribution)
noise_prior = torch.distributions.uniform.Uniform(0, 1)

# VERY temporary generator
class BadGenerator(object):
    def __init__(self):
        pass

    # just use the prior noise dist
    def __call__(self, x):
        return x

D = DiscriminatorNetwork(input_dim=1, output_dim=1, hidden_dim=20, n_hidden_layers=1)
G = GeneratorNetwork(intput_dim=1, output_dim=1, hidden_dim=20, n_hidden_layers=1)

N = 1000 # number of epochs to train D + G
k = 10 # train the discriminator for k steps each epoch
m = 50 # minibatch size

# loss for the discriminator 
def compute_discriminator_loss(y_train, y_gen):
    return -torch.mean(torch.log(y_train) + torch.log(1 - y_gen))

# loss for the generator, note that we change the objective as described in the original paper
def compute_generator_loss(y_gen):
    return torch.mean(torch.log(y_gen))

discriminator_optim = torch.optim.Adam(params=D.parameters())
# we don't train the generator for this little test 
# generator_optim = torch.optim.Adam(

plt.figure(figsize=(20,10))
# training epochs
for i in range(N):
    # iterations of discriminator training
    for j in range(k):
        # sample a minibatch from the training data and the generator
        perm = torch.randperm(n_train)
        x_train_mb = x_train[perm[:m]]
        z_mb = noise_prior.sample((m, 1))
        # saying that the generator just keeps making 0 or 1
        # z_mb = torch.randint(0, 2, size=(m, 1)).float()
        # z_mb = torch.full_like(x_train_mb, 0.2)
        x_gen_mb = G(z_mb)

        # discriminator predictions for the training data
        y_train_mb = D(x_train_mb)
        # discriminator predictions for the generator data
        y_gen_mb = D(x_gen_mb)

        # compute the discriminator loss and backprop
        discriminator_optim.zero_grad()
        d_loss = compute_discriminator_loss(y_train_mb, y_gen_mb)
        d_loss.backward()
        discriminator_optim.step()
    # one per epoch we train the generator
    generator_optim.zero_grad()
    g_loss = compute_generator_loss(y_gen
    # plotting the discriminator curve during training
    if i % 100 == 0:
        print(f"Loss at epoch {i} is {d_loss}")
plt.plot(x_eval, D(x_eval).detach(), label=f"After epoch {i}")
z = torch.full_like(x_train, 0.8)
x_gen = G(z)
plt.hist(x_train.detach().numpy(), bins=30, density=True, label="Training data distribution")
# plt.hist(x_gen.detach().numpy(), bins=30, density=True, label="Training data distribution")
plt.legend()
plt.show()
        
