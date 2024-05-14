import torchvision
from coco2014 import CocoDataset2014
import pathlib
import torch
import math
import sys

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
    BATCH_SIZE = 8;
    lr = 0.001;
    print(f"Using device: {device}");
    # no pretrained layers
    # pretrained backbone on ImageNet is fine 
    # 2 classes: Object, and No Object --> text or no text
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, weights_backbone=True, num_classes=2).to(device);    
    # load weight file if possible
    if(W_PATH.stat().st_size != 0):
        model.load_state_dict(torch.load(str(W_PATH.absolute())));
        print("Successfully loaded model's previous state!");
    model.train();

    # stochastic gradient descent for model parameters
    params = [p for p in model.parameters() if p.requires_grad];
    optimizer = torch.optim.SGD(params, lr);

    epochs = 1;
    dataset = CocoDataset2014("cocotext.v2.json", "train2014",device=device);

    print(f"Number of images in training set: {len(dataset.train_imgs)}");
    dataset_generator = dataset.input_loader(BATCH_SIZE);
    for epoch in range(epochs):
        print(f"Epoch {epoch}: ");
        for i in range(int(len(dataset.train_imgs) / BATCH_SIZE) + 1):
            img, target = next(dataset_generator);
            output = model(img.to(device), target);
            # print the loss
            print(f"Classification loss: {output['classification']:.2f}\t[{(i+1) * BATCH_SIZE} / {len(dataset.train_imgs)}]");
            print(f"Bounding Box loss: {output['bbox_regression']:.2f}");

            losses = sum(loss for loss in output.values());
            if not math.isfinite(losses):
                print(f"Loss is {losses}, stopping training");
                sys.exit(1);

            optimizer.zero_grad();
            losses.backward();
            optimizer.step();
    
    
    torch.save(model.state_dict(), str(W_PATH.absolute()));



if(__name__ == "__main__"):
    main();

