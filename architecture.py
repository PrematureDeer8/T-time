
# reimplementation of https://arxiv.org/pdf/1708.02002
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.ops


class T_time(nn.Module):
    def __init__(self):
        super(T_time, self).__init__();
        # feature pyramid network
        # out channels will be 256
        self.fpn = torchvision.ops.FeaturePyramidNetwork([512,1024,2048], 256);
        self.features = {};
        # resnet backbone
        self.backbone = resnet50(weights="DEFAULT");
    
    def forward(self, x):
        x = self.backbone.conv1(x);
        x = self.backbone.bn1(x);
        x = self.backbone.relu(x);
        x = self.backbone.maxpool(x);
        x = self.backbone.layer1(x);

        # these features will go into the FPN
        self.features["feat2"] = self.backbone.layer2(x);
        self.features["feat3"] = self.backbone.layer3(self.features["feat2"]);
        self.features["feat4"] = self.backbone.layer4(self.features["feat3"] );
        
        self.fpn(self.features);

    

