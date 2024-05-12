import torchvision.models.detection.retinanet_resnet50_fpn as retinanet
import pathlib
import json

def cocotextloader(file_path):
    fp = pathlib.Path(file_path);
    if(not fp.exists()):
        return print(f"File path ({str(fp.absolute())}) does not exist!");
    with fp.open() as file:
        annotations = json.load(file);
    return annotations;


def main():
    # no pretrained layers
    # pretrained backbone on ImageNet is fine 
    # 2 classes: Object, and No Object --> text or no text
    model = retinanet(weights=None, weights_backbone=True, num_classes=2);
    model.train();

    epoch = 1;

    


if(__name__ == "__main__"):
    main();

