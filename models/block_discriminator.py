#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Local imports
from .tgcn import ConvTemporalGraphical

class st_gcn(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                graph=None,
                lvl=3,
                dropout=0,
                residual=True,
                dw_s=False, dw_t=64,
                device=None):
        super(st_gcn, self).__init__()
        self.device = device
        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.dw_s, self.dw_t = graph, lvl, dw_s, dw_t
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                        kernel_size[1][lvl])

        self.tcn = nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            )


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=(stride, 1)
                    )


        self.l_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, A):
        x = x.to(self.device)
        A = A.to(self.device)

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x    = self.tcn(x) + res

        x = self.downsample_s(x) if self.dw_s else x
        
        x = F.interpolate(x, size=(self.dw_t,x.size(-1)))  # Exactly like nn.Upsample

        return self.l_relu(x), A


    def downsample_s(self, tensor):
        keep = self.graph.map[self.lvl+1][:,1]

        return tensor[:,:,:,keep]