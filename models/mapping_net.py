#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Mapping_Net(nn.Module):
    def __init__(self, device, latent=1024, mlp=4):
        super(Mapping_Net, self).__init__()
        self.device = device

        layers = []
        for i in range(mlp):
            linear = nn.Linear(latent, latent)
            linear.weight.data.normal_()
            linear.bias.data.zero_()
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))

        self.mlp = nn.Sequential(*layers)
        self.mlp.to(device)

    def forward(self, x):
        x = x.to(self.device)
        return self.mlp(x)