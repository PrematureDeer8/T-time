from retinanet import RetinaNet
from dataset import CTWDataset
import pathlib
import torch
import math
import sys
from torch.utils import data
from loss import FocalLoss
import numpy as np
import sys
import math

#weights path
W_PATH = pathlib.Path(".") / "architecture" / "state.pt";
W_PATH.parent.mkdir(parents=True, exist_ok=True);
W_PATH.touch(exist_ok=True);

# check device
if(torch.cuda.is_available()):
    # torch.set_default_device("cuda");
    device = ("cuda");
elif(torch.backends.mps.is_available()):
    # torch.set_default_device("mps");
    device = ("mps");
else:
    device = ("cpu");

def main():
    BATCH_SIZE = 16;
    lr = 0.01;
    print(f"Using device: {device}");

    if(sys.platform == "darwin"):
        dataset = CTWDataset();
    else:
        dataset = CTWDataset(
            annotation_file=str(pathlib.Path.home() / "Downloads/archive/ctw1500_train_labels"),
            img_folder=str(pathlib.Path.home() / "Downloads/archive/train_images")
        );

    # no pretrained layers
    # pretrained backbone on ImageNet is fine 
    # 2 classes: Object, and No Object --> text or no text
    model = RetinaNet().to(device);
    # load weight file if possible
    if(W_PATH.stat().st_size != 0):
        model.load_state_dict(torch.load(str(W_PATH.absolute()), map_location=device));
        print("Successfully loaded model's previous state!");
    model.train();

    # stochastic gradient descent for model parameters
    params = [p for p in model.parameters() if p.requires_grad];
    optimizer = torch.optim.SGD(params, lr);

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_func = FocalLoss();

    epochs = 25;

    print(f"Number of images in training set: {len(dataset)}");
    train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False);
    for epoch in range(epochs):
        epoch_loss = [];
        total_loss = 0;
        for batch, (x_train, y_train) in enumerate(train_loader):
            classification,regression, anchors =  model(x_train.to(device));
            cls_loss, reg_loss = loss_func(classification.to(device),regression,anchors.to(device), y_train.to(device));
            
            # cls_loss = cls_loss.mean();
            # reg_loss = reg_loss.mean();
            loss = cls_loss + reg_loss;

            if(not math.isfinite(loss)):
                print("Loss is not finite! Stopping training!");
                sys.exit(0);
            optimizer.zero_grad();

            loss.backward();

            torch.nn.utils.clip_grad_norm_(params, 0.1);

            optimizer.step();

            epoch_loss.append(float(loss));

            total_loss += loss.item();
            if(batch % 3*(BATCH_SIZE) == 0):
                loss, current = loss.item(), (batch + 1) * len(x_train);
                print(f"class loss: {float(cls_loss):>7f}\tregresion loss: {float(reg_loss):>7f} \tloss: {loss:>7f} [{current:>5d}/{len(train_loader.dataset):>5}]");
            del cls_loss
            del reg_loss
        print(f"Epoch {epoch}: {total_loss / len(train_loader)}");
        scheduler.step(np.mean(epoch_loss));
    
    
    torch.save(model.state_dict(), str(W_PATH.absolute()));



if(__name__ == "__main__"):
    main();

