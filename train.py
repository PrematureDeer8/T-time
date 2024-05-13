import torchvision
from coco2014 import CocoDataset2014
import pathlib
import torch

#weights path
W_PATH = pathlib.Path(".") / "architecture" / "state.pt";
W_PATH.parent.mkdir(parents=True, exist_ok=True);
W_PATH.touch(exist_ok=True);

# check device
if(torch.cuda.is_available()):
    device = ("cuda");
elif(torch.backends.mps.is_available()):
    device = ("mps");
else:
    device = ("cpu");

def main():
    print(f"Using device: {device}");
    # no pretrained layers
    # pretrained backbone on ImageNet is fine 
    # 2 classes: Object, and No Object --> text or no text
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, weights_backbone=True, num_classes=2).to(device);    
    # load weight file if possible
    if(W_PATH.stat().st_size != 0):
        model.load_state_dict(torch.load(str(W_PATH.absolute())));
    model.train();

    epochs = 1;
    dataset = CocoDataset2014("cocotext.v2.json", "train2014");

    print(f"Number of images in training set: {len(dataset.train_imgs)}");
    dataset_generator = dataset.input_loader(64);
    for epoch in range(epochs):
        img, target = next(dataset_generator);
        _ = model(img.to(device), target);
    
    
    torch.save(model.state_dict(), str(W_PATH.absolute()));



if(__name__ == "__main__"):
    main();

