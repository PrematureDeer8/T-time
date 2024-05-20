from retinanet import RetinaNet
from dataset import CTWDataset
import pathlib
import torch
import math
import sys
from torch.utils import data
from loss import FocalLoss

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
    BATCH_SIZE = 16;
    lr = 0.01;
    print(f"Using device: {device}");
    # no pretrained layers
    # pretrained backbone on ImageNet is fine 
    # 2 classes: Object, and No Object --> text or no text
    model = RetinaNet();
    # load weight file if possible
    if(W_PATH.stat().st_size != 0):
        model.load_state_dict(torch.load(str(W_PATH.absolute())));
        print("Successfully loaded model's previous state!");
    model.train();

    # stochastic gradient descent for model parameters
    params = [p for p in model.parameters() if p.requires_grad];
    optimizer = torch.optim.SGD(params, lr);

    loss_func = FocalLoss();

    epochs = 1;
    dataset = CTWDataset();

    print(f"Number of images in training set: {len(dataset)}");
    train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True);
    for epoch in range(epochs):
        total_loss = 0;
        for batch, (x_train, y_train) in enumerate(train_loader):
            classification,regression, anchors =  model(x_train);
            cls_loss, reg_loss = loss_func(classification,regression,anchors, y_train);
            
            loss = cls_loss + reg_loss;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item();
            if(batch % 10 == 0):
                loss, current = loss.item(), (batch + 1) * len(x_train);
                print(f"loss: {loss:>7f} [{current:>5d}/{len(train_loader.dataset):>5}]");
        print(f"Epoch {epoch}: {total_loss / len(train_loader)}");
    
    
    torch.save(model.state_dict(), str(W_PATH.absolute()));



if(__name__ == "__main__"):
    main();

