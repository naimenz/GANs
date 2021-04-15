"""
Training using my new setup with a GAN object
"""

from gan import GenerativeAdversarialNetwork, MLPDiscriminator, MLPGenerator
from gan import DataProvider
from load_mnist import load_mnist

import torch
import matplotlib.pyplot as plt
import numpy as np

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

# evaluation grid
n_eval = 100
x_eval = torch.linspace(0, 1, n_eval).reshape(-1, 1)

x_train = load_mnist('mnist')[0] / 255 # dividing by 255 to get between 0 and 1
data_provider = DataProvider(x_train.astype(np.float32))
noise_prior = torch.distributions.uniform.Uniform(0, 1)

G = MLPGenerator(noise_prior, input_dim=10, output_dim=784, hidden_dim=256, n_hidden_layers=3, lr=1e-3)
D = MLPDiscriminator(input_dim=784, hidden_dim=256, n_hidden_layers=3, lr=1e-3)

N = 1000 # number of epochs
# k = 200 # number of updates to the discriminator each epoch
k = 100 # they use 1 in the paper
m = 128 # number of samples per minibatch
gan = GenerativeAdversarialNetwork(G, D, data_provider, m, k)

# training loop
for i in range(N):
    # DEBUG 
    d_loss, g_loss = gan.train_epoch(epoch=i)
    print(f"At epoch {i}, discriminator loss is {d_loss:0.4f} and generator loss is {g_loss:0.4f}")
    if i % 100 == 0:
        gan.G.save('G_params.pt')
        gan.D.save('D_params.pt')

# gan.G.restore('G_params.pt')
# gan.D.restore('D_params.pt')

# plotting!
x_gen = gan.G.sample(10, training=False)
plt.imshow((x_gen[0] - x_gen[1]).detach().numpy().reshape(28,28))
plt.colorbar()
plt.show()
for i in range(5):
    plt.imshow(x_gen[i].detach().numpy().reshape(28, 28))
    plt.colorbar()
    plt.savefig(f'mnist_example{i}.pdf')
    plt.show()

# plt.figure(figsize=(20,10))
# plt.hist(x_train.detach().numpy(), bins=50, density=True, label="Training data distribution")
# plt.hist(x_gen.detach().numpy(), bins=50, density=True, label="Generator data distribution")
# plt.plot(x_eval, gan.D(x_eval).detach().numpy(), label=f"Discriminator prediction curve")
plt.show()
