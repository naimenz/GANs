"""
Testing what I've written for the discriminator and generator
"""

from gan import DiscriminatorNetwork, GeneratorNetwork
import torch
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
torch.manual_seed(SEED)

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

n_train = 20000
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


N = 1000 # number of epochs to train D + G
k = 100#20 # train the discriminator for k steps each epoch
m = 10 # minibatch size
mg = 100 # minibatch for generator size

# loss for the discriminator 
def compute_discriminator_loss(y_train, y_gen):
    return -torch.mean(torch.log(y_train) + torch.log(1 - y_gen))

# loss for the generator, note that we change the objective as described in the original paper
def compute_generator_loss(y_gen):
    return -torch.mean(torch.log(y_gen))
    # return torch.mean(1-torch.log(y_gen))

D = DiscriminatorNetwork(input_dim=1, output_dim=1, hidden_dim=100, n_hidden_layers=1)
G = GeneratorNetwork(input_dim=1, output_dim=1, hidden_dim=100, n_hidden_layers=1)

discriminator_optim = torch.optim.Adam(params=D.parameters())
generator_optim = torch.optim.Adam(params=G.parameters())
# we don't train the generator for this little test 
plt.figure(figsize=(20,10))
# training epochs
for i in range(N):
    # iterations of discriminator training
    d_losses = []
    for j in range(k):
        # sample a minibatch from the training data and the generator
        perm = torch.randperm(n_train)
        x_train_mb = x_train[perm[:m]]

        z_mb = noise_prior.sample((m, 1))
        x_gen_mb = G(z_mb)

        # discriminator predictions for the training data
        y_train_mb = D(x_train_mb)
        # discriminator predictions for the generator data
        y_gen_mb = D(x_gen_mb)

        # compute the discriminator loss and backprop
        discriminator_optim.zero_grad()
        d_loss = compute_discriminator_loss(y_train_mb, y_gen_mb)
        d_loss.backward()
        d_losses.append(d_loss.item())
        discriminator_optim.step()

    # one per epoch we train the generator on a new mb
    z_mb = noise_prior.sample((mg, 1))
    y_gen_mb = D(G(z_mb))

    generator_optim.zero_grad()
    g_loss = compute_generator_loss(y_gen_mb)
    g_loss.backward()
    generator_optim.step()

    # plotting the discriminator curve during training
    if i % 10 == 0:
        print(f"Discriminator loss at epoch {i} is {np.mean(d_losses)}")
        print(f"Generator loss at epoch {i} is {g_loss}")
        
z = noise_prior.sample((n_train, 1))
x_gen = G(z)
plt.hist(x_train.detach().numpy(), bins=50, density=True, label="Training data distribution")

# plt.plot(x_eval, G(x_eval).detach(), label=f"Generator after epoch {i}")
plt.hist(x_gen.detach().numpy(), bins=50, density=True, label="Generator data distribution")
plt.plot(x_eval, D(x_eval).detach(), label=f"Discriminator after epoch {i}")
# plt.hist(z.detach().numpy(), bins=30, density=True, label="Noise prior data distribution")
plt.legend()
plt.show()
