from torch import nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
from subnet import Subnet
from anchors import Anchors

class RetinaNet(nn.Module):
    def __init__(self,feature_dim=256) -> None:
        super(RetinaNet, self).__init__();
        # resnet50
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT);

        # Feature pyramid Network (FPN) referenced:
        # https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/model.py#L19
        self.conv1x1_C5 = nn.Conv2d(2048, feature_dim, kernel_size=1);
        self.upsample_P5 = nn.Upsample(scale_factor=2, mode="nearest");
        self.conv1x1_C4 = nn.Conv2d(1024, feature_dim, kernel_size=1);
        self.upsample_P4 = nn.Upsample(scale_factor=2, mode="nearest");
        self.conv1x1_C3 = nn.Conv2d(512, feature_dim, kernel_size=1);
        self.upsample_P3 = nn.Upsample(scale_factor=2, mode="nearest");
        self.conv1x1_C2 = nn.Conv2d(256, feature_dim, kernel_size=1);
        self.conv3x3_P5 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        self.conv3x3_P4 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        self.conv3x3_P3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        self.conv3x3_P2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        # for P6
        self.seq_P6 = nn.Sequential(nn.ReLU(), nn.Conv2d(feature_dim,feature_dim, kernel_size=3,stride=2, padding=1));

        # subnets
        self.classification = Subnet("classification", 256);
        self.regression = Subnet("regression", 256);

        # anchors
        self.anchors = Anchors();
    

    def forward(self, img_batch):

        x = self.backbone.conv1(img_batch);
        x = self.backbone.bn1(x);
        x = self.backbone.relu(x);
        x = self.backbone.maxpool(x);

        # bottom up pathway
        C2 = self.backbone.layer1(x);
        C3 = self.backbone.layer2(C2);
        C4 = self.backbone.layer3(C3);
        C5 = self.backbone.layer4(C4);

        # top down pathway
        P5 = self.conv1x1_C5(C5);
        P4 = self.upsample_P5(P5) + self.conv1x1_C4(C4);
        P3 = self.upsample_P4(P4) + self.conv1x1_C3(C3);
        P2 = self.upsample_P3(P3) + self.conv1x1_C2(C2);
        # 3x3 convolution to reduce aliasing effect from upsampling
        P5 = self.conv3x3_P5(P5);
        P4 = self.conv3x3_P4(P4);
        P3 = self.conv3x3_P3(P3);
        P2 = self.conv3x3_P2(P2);
        P6 = self.seq_P6(P5);

        self.features = [P2, P3, P4, P5, P6];
        self.regr = torch.cat([self.regression(feature) for feature in self.features], dim=1);
        self.cls = torch.cat([self.classification(feature) for feature in self.features], dim=1);

        self.candidate_anchors = self.anchors(img_batch);