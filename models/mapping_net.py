#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Mapping_Net(nn.Module):
    def __init__(self, latent=1024, mlp=4):
        super(Mapping_Net, self).__init__()

        layers = []
        for i in range(mlp):
            linear = nn.Linear(latent, latent)
            linear.weight.data.normal_()
            linear.bias.data.zero_()
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)