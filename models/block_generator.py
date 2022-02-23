#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Local imports
from .noise_injection import NoiseInjection
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
                bn=True,
                residual=True,
                up_s=False, 
                up_t=64, 
                tan=False,
                device=None):
        super(st_gcn, self).__init__()

        self.device = device

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.up_s, self.up_t, self.tan = graph, lvl, up_s, up_t, tan
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                        kernel_size[1][lvl])

        tcn = [nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            )]
        
        tcn.append(nn.BatchNorm2d(out_channels)) if bn else None

        self.tcn = nn.Sequential(*tcn)


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.noise = NoiseInjection(out_channels)

        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh   = nn.Tanh()

    def forward(self, x, A):
        x = x.to(self.device)
        A = A.to(self.device)

        x = self.upsample_s(x) if self.up_s else x
        
        x = F.interpolate(x, size=(self.up_t,x.size(-1)))  # Exactly like nn.Upsample

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x    = self.tcn(x) + res
        
        # Noise Inject
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=self.device)
        x     = self.noise(x, noise)

        return self.tanh(x) if self.tan else self.l_relu(x), A

    
    def upsample_s(self, tensor):

        ids  = []
        mean = []
        for umap in self.graph.mapping[self.lvl]:
            ids.append(umap[0])
            tmp = None
            for nmap in umap[1:]:
                tmp = torch.unsqueeze(tensor[:, :, :, nmap], -1) if tmp == None else torch.cat([tmp, torch.unsqueeze(tensor[:, :, :, nmap], -1)], -1)

            mean.append(torch.unsqueeze(torch.mean(tmp, -1) / (2 if self.lvl==2 else 1), -1))

        for i, idx in enumerate(ids): tensor = torch.cat([tensor[:,:,:,:idx], mean[i], tensor[:,:,:,idx:]], -1)


        return tensor