from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class RetinaNet(nn.Module):
    def __init__(self,feature_dim=256) -> None:
        super(RetinaNet, self).__init__();
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT);
        # Feature pyramid Network (FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest");
        # reduce channel dimensions of coarser feature map
        self.conv1x1_2048 = nn.Conv2d(2048, 1024,(1,1));
        self.conv1x1_1024 = nn.Conv2d(1024, 512, (1,1));
        self.conv1x1_512 = nn.Conv2d(512, 256, (1,1));
        # 3x3 convolution for channel dimension reduction
        self.conv3x3_1024 = nn.Conv2d(1024, feature_dim, (3,3));
        self.conv3x3_512 = nn.Conv2d(512, feature_dim, (3,3));
        self.conv3x3_256 = nn.Conv2d(256, feature_dim, (3,3));
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

        # top down pathway
        P4 = self.conv3x3_1024(C4 + self.conv1x1_2048(self.upsample(C5)));
        P3 = self.conv3x3_512(C3 + self.conv1x1_1024(self.upsample(C4)));
        P2 = self.conv3x3_256(C2 + self.conv1x1_512(self.upsample(C3)));