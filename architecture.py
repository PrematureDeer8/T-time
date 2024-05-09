
# reimplementation of https://dl-acm-org.ezproxy.lib.purdue.edu/doi/pdf/10.1145/3356728
import torch
from torch import nn
from torchvision.models import vgg16


class T_time(nn.Module):
    def __init__(self):
        super(T_time, self).__init__();
        # standard vgg16
        vgg = vgg16();
        # get rid of last pooling layer (also going to forgo the Adaptive Pooling)
        # drop the classification layer (FC layers)
        backbone = vgg.features[:-1];
        # stage 1 and 2
        self.stage_1_2 = backbone[:4];
        # stage 3
        self.stage3 = backbone[4:9];
        # stage 4
        self.stage4 = backbone[9:16];
        # stage 5
        self.stage5 = backbone[16:];
    def forward(self, x):
        x = self.stage_1_2(x); # --> pass this to MCFE
        y1 = self.stage3(x); # --> pass this to MCFE
        y2 = self.stage4(y1); # --> pass this to MCFE
        self.stage5(y2);

    def MCFE(self, x):
        raise NotImplementedError;