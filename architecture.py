
# reimplementation of https://doi.org/10.1145/3356728
import torch
from torch import nn
from torchvision.models import vgg16


class T_time(nn.Module):
    def __init__(self):
        super(T_time, self).__init__();
        # parameters of vgg are initialized by VGG-16 weights
        vgg = vgg16(weights="DEFAULT");
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
        # MCFE dilated convolution 
        dilations = [1,3,5,7]; 
        self.MCFE1 = [nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3),padding=dilation, dilation=dilation) for dilation in dilations];
        self.MCFE2 = [nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3),padding=dilation, dilation=dilation) for dilation in dilations];
        self.MCFE3 = [nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3,3),padding=dilation, dilation=dilation) for dilation in dilations];
    
    def forward(self, x):
        x = self.stage_1_2(x); 
        f3 = self.stage3(x); # --> pass this to MCFE
        self.MCFE(f3, code=1);
        f4 = self.stage4(f3); # --> pass this to MCFE
        self.MCFE(f4, code=2);
        f5 = self.stage5(f4); # --> pass this to MCFE
        self.MCFE(f5, code=3);

    def MCFE(self,feature, code=1):
        if(code == 1):
            return torch.concat([self.MCFE1[0](feature),
                                self.MCFE1[1](feature),
                                self.MCFE1[2](feature),
                                self.MCFE1[3](feature)]);
        elif(code == 2):
            return torch.concat([self.MCFE2[0](feature),
                                self.MCFE2[1](feature),
                                self.MCFE2[2](feature),
                                self.MCFE2[3](feature)]);
        elif(code == 3):
            return torch.concat([self.MCFE3[0](feature),
                                self.MCFE3[1](feature),
                                self.MCFE3[2](feature),
                                self.MCFE3[3](feature)]);
        else:
            print("No such code {}!".format(code))
            raise IndexError;
    

