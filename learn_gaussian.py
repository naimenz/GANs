"""
Training using my new setup with a GAN object
"""

from gan import GenerativeAdversarialNetwork, MLPDiscriminator, MLPGenerator
from gan import DataProvider

import torch
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# evaluation grid
n_eval = 1000
x_eval = torch.linspace(0, 1, n_eval).reshape(-1, 1)

# training data
n_train = 20000
mean, var = 0.5, 0.1
x_train = torch.normal(mean, var, (n_train, 1))
data_provider = DataProvider(x_train.numpy())

noise_prior = torch.distributions.uniform.Uniform(0, 1)

G = MLPGenerator(noise_prior, input_dim=1, output_dim=1, hidden_dim=100, n_hidden_layers=1, lr=1e-3)
D = MLPDiscriminator(input_dim=1, hidden_dim=100, n_hidden_layers=1, lr=1e-3)

N = 2000 # number of epochs
k = 100 # number of updates to the discriminator each epoch
m = 200 # number of samples per minibatch
gan = GenerativeAdversarialNetwork(G, D, data_provider, m, k)

# training loop
for i in range(N):
    g_loss, d_loss = gan.train_epoch()
    if i % 25 == 0:
        print(f"At epoch {i}, discriminator loss is {d_loss:0.4f} and generator loss is {g_loss:0.4f}")

# plotting!
x_gen = gan.G.sample(n_train)

plt.figure(figsize=(20,10))
plt.hist(x_train.detach().numpy(), bins=50, density=True, label="Training data distribution")
plt.hist(x_gen.detach().numpy(), bins=50, density=True, label="Generator data distribution")
plt.plot(x_eval, gan.D(x_eval).detach().numpy(), label=f"Discriminator prediction curve")
plt.savefig('gaussian_dist.pdf')
plt.show()
