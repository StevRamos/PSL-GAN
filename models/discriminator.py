#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Third party imports
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#Local imports
from utils.graph_psl import GraphPSL
from .block_discriminator import st_gcn


class Discriminator(nn.Module):
    
    def __init__(self, in_channels, 
                n_classes, t_size, 
                latent, edge_importance_weighting=True, 
                device=None, **kwargs):
        super(Discriminator, self).__init__()
        self.device = device
        # load graph
        self.graph = GraphPSL()
        self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False).to(device) for Al in self.graph.As]

        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [3 for _ in self.A]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        self.t_size          = t_size

        #kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels+n_classes, 32, kernel_size, 1, graph=self.graph, lvl=0, dw_s=True, dw_t=t_size, residual=False, device=device, **kwargs),
            st_gcn(32, 64, kernel_size, 1, graph=self.graph, lvl=1, dw_s=True, dw_t=math.ceil(t_size/1.25), device=device, **kwargs),
            st_gcn(64, 128, kernel_size, 1, graph=self.graph, lvl=2, dw_s=False, dw_t=math.ceil(t_size/1.7), device=device, **kwargs),
            st_gcn(128, 256, kernel_size, 1, graph=self.graph, lvl=2, dw_s=True, dw_t=math.ceil(t_size/2.5), device=device, **kwargs),
            st_gcn(256, 512, kernel_size, 1, graph=self.graph, lvl=3, dw_s=True, dw_t=math.ceil(t_size/5),  device=device, **kwargs),
            st_gcn(512, latent, kernel_size, 1, graph=self.graph, lvl=4, dw_s=False, dw_t=1, device=device, **kwargs),
        ))


        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A[i.lvl].size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.label_emb = nn.Embedding(n_classes, n_classes)

        # fcn for prediction
        self.fcn = nn.Linear(latent, 1)

    def forward(self, x, labels):
        x = x.to(self.device)
        labels = labels.to(self.device)
        
        N, C, T, V = x.size()


        c = self.label_emb(labels)
        c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, T, V)

        x = torch.cat((c, x), 1)
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)

        
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)

        # prediction
        validity = self.fcn(x)

        return validity