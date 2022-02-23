import os

import torch.nn.init as init
import torch
import wandb
import numpy as np

def init_optimizer(generator, discriminator, lr, b1=0.5, b2=0.999):
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    return opt_g, opt_d


def save_weights(generator, discriminator, path, use_wandb):
    torch.save(generator.state_dict(), os.path.join(path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

    if use_wandb:
        wandb.save(os.path.join(path, '*.pth'),
                    base_path='/'.join(path.split('/')[:-2]))   