#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection, self).__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise