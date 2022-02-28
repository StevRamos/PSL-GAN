#Taken from https://github.com/DegardinBruno/Kinetic-GAN

import sys

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#Local imports
from utils.graph_psl import GraphPSL
from .mapping_net import Mapping_Net
from .block_generator import st_gcn

class Generator(nn.Module):
    
    def __init__(self, in_channels, 
                out_channels, n_classes, 
                t_size, mlp_dim=4, edge_importance_weighting=True, 
                device=None, **kwargs):
        super(Generator, self).__init__()
        self.device = device
        # load graph
        self.graph = GraphPSL()
        self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False).to(device) for Al in self.graph.As]

        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [3 for i, _ in enumerate(self.A)]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        self.t_size          = t_size

        self.mlp = Mapping_Net(self.device, in_channels+n_classes, mlp_dim)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels+n_classes, 512, kernel_size, 1, graph=self.graph, lvl=4, bn=False, residual=False, up_s=False, up_t=1,device=self.device, **kwargs),
            st_gcn(512, 256, kernel_size, 1, graph=self.graph, lvl=3, up_s=True, up_t=2, device=self.device, **kwargs),
            st_gcn(256, 128, kernel_size, 1, graph=self.graph, lvl=2, bn=False, up_s=True, up_t=2, device=self.device, **kwargs),
            st_gcn(128, 64, kernel_size, 1, graph=self.graph, lvl=2, up_s=False, up_t=4, device=self.device, **kwargs),
            st_gcn(64, 32, kernel_size, 1, graph=self.graph, lvl=1, bn=False, up_s=True, up_t=6, device=self.device, **kwargs),
            st_gcn(32, out_channels, kernel_size, 1, graph=self.graph, lvl=1, up_s=False, up_t=8, device=self.device, **kwargs),
            st_gcn(out_channels, out_channels, kernel_size, 1, graph=self.graph, lvl=0, bn=False, up_s=True, up_t=t_size, tan=True, device=self.device, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A[i.lvl].size()))
                for i in self.st_gcn_networks
            ])
            
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.edge_importance.to(self.device)
        self.label_emb = nn.Embedding(n_classes, n_classes).to(self.device)
        

    def forward(self, x, labels, trunc=None):
        x = x.to(self.device)
        labels = labels.to(self.device)

        c = self.label_emb(labels)

        x = torch.cat((c, x), -1)

        w = []
        for i in x:
            w = self.mlp(i).unsqueeze(0) if len(w)==0 else torch.cat((w, self.mlp(i).unsqueeze(0)), dim=0)

        w = self.truncate(w, 1000, trunc) if trunc is not None else w  # Truncation trick on W

        x = w.view((*w.shape, 1, 1))

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)

        return x

    def truncate(self, w, mean, truncation):  # Truncation trick on W
        t = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (mean, *w.shape[1:])))).to(self.device)
        w_m = []
        for i in t:
            w_m = self.mlp(i).unsqueeze(0) if len(w_m)==0 else torch.cat((w_m, self.mlp(i).unsqueeze(0)), dim=0)

        m = w_m.mean(0, keepdim=True)

        for i,_ in enumerate(w):
            w[i] = m + truncation*(w[i] - m)

        return w

