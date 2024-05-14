from torch import nn
from torchvision.models import resnet50

class RetinaNet(nn.Module):
    def __init__(self) -> None:
        super(RetinaNet, self).__init__();
        self.backbone = resnet50(weights="DEFAULT");
        # use of stride of 2 for the conv2 (layers: 2,3,4)
        # source: https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/utils.py
        self.backbone.layer2.conv2.stride = 2;
        self.backbone.layer3.conv2.stride = 2;
        self.backbone.layer4.conv2.stride = 2;

    def forward(self, x):

        x = self.backbone.conv1(x);
        x = self.backbone.bn1(x);
        x = self.backbone.relu(x);
        x = self.backbone.maxpool(x);

        # bottom up pathway
        C2 = self.backbone.layer1(x);
        C3 = self.backbone.layer2(C2);
        C4 = self.backbone.layer3(C3);
        C5 = self.backbone.layer4(C4);