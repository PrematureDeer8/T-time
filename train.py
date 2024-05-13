import torchvision.models.detection.retinanet_resnet50_fpn_v2 as retinanet



def main():
    # no pretrained layers
    # pretrained backbone on ImageNet is fine 
    # 2 classes: Object, and No Object --> text or no text
    model = retinanet(weights=None, weights_backbone=True, num_classes=2);
    model.train();

    epoch = 1;

    


if(__name__ == "__main__"):
    main();

