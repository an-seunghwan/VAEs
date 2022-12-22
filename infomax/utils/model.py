#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
#%%
class Discriminator(nn.Module):
    def __init__(self, z_dim=2, device='cpu'):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(784 + z_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        ).to(device)

    def forward(self, x, z):
        x = x.view(-1, 784)
        x = torch.cat((x, z), dim=1)
        return self.net(x)
#%%
class VAE(nn.Module):
    def __init__(self, z_dim=2, device='cpu'):
        super(VAE, self).__init__()
        
        self.z_dim = z_dim
        self.device = device
        
        self.encode = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2 * z_dim),
        ).to(device)
        
        self.decode = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 784),
            nn.Sigmoid(),
        ).to(device)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # new(): The type and device of this Tensor match the existing Tensor and have no contents.
        # normal_(): Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
        eps = std.data.new(std.size()).normal_()
        return mu + eps.mul(std)
    
    def generation(self, x):
        gen_z = Variable(torch.randn(100, self.z_dim), requires_grad=False).to(self.device)
        gen_z = gen_z.to(self.device)
        return self.decode(gen_z).view(x.size())
    
    def get_posterior(self, x):
        stats = self.encode(x.view(-1, 784))
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        return mu, logvar
    
    def forward(self, x):
        mu, logvar = self.get_posterior(x)
        z = self.reparametrize(mu, logvar)
        xhat = self.decode(z).view(x.size())
        return xhat, mu, logvar, z
#%%
def main():
    config = {
        'z_dim': 2,
    }
    
    x = torch.randn(10, 1, 28, 28)
    model = VAE(config["z_dim"])
    discriminator = Discriminator(config["z_dim"])
    xhat, mu, logvar, z = model(x)
    t = discriminator(xhat, z)
    print(model)
    
    assert xhat.shape == x.shape
    assert mu.shape == (x.size(0), config["z_dim"])
    assert logvar.shape == (x.size(0), config["z_dim"])
    assert z.shape == (x.size(0), config["z_dim"])
    assert t.shape == (x.size(0), 1)
    
    print('Model test pass!')
#%%
if __name__ == "__main__":
    main()
#%%