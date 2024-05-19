from torch import nn
import math

pi = 0.01

class Subnet(nn.Module):
    def __init__(self,type, num_features_in, anchors=9, feature_dim=256, num_classes=1):
        super(Subnet, self).__init__();
        self.type = type;
        self.anchors = anchors;
        self.num_classes = num_classes;

        self.conv1 = nn.Conv2d(num_features_in, feature_dim, kernel_size=3, padding=1);
        self.act1 = nn.ReLU();

        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        self.act2 = nn.ReLU();

        self.conv3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        self.act3 = nn.ReLU();

        self.conv4 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1);
        self.act4 = nn.ReLU();

        if(self.type.lower() == 'regression'):
            self.output = nn.Conv2d(feature_dim, anchors * 4, kernel_size=3, padding=1);
            self.type = 0;
        elif(self.type.lower() == "classification"):
            self.output = nn.Conv2d(feature_dim, anchors * self.num_classes, kernel_size=3, padding=1, bias=math.log10((1-pi)/pi));
            self.output_act = nn.Sigmoid();
            self.type = 1;
        else:
            raise TypeError(f"{self.type} is not a option! Choose between regression or classification");
    
    def forward(self, x):
        out = self.conv1(x);
        out = self.act1(out);

        out = self.conv2(out);
        out = self.act2(out);

        out = self.conv3(out);
        out = self.act3(out);

        out = self.conv4(out);
        out = self.act4(out);

        out = self.output(out);
        if(self.type):
            out = self.output_act(out);
            out1 = out.permute(0,2,3,1);

            batch_size, width, height, channels = out1.shape;

            out2 = out1.view(batch_size, width, height, self.anchors, self.num_classes);
            return out2.contiguous().view(x.shape[0], -1, self.num_classes);
        else:
            out = out.permute(0,2,3,1);

            return out.contiguous().view(out.shape[0], -1, 4);
            

