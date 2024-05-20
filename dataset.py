import pathlib
import json
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
import torch
import xml.etree.ElementTree as ET


class CTWDataset(Dataset):
    def __init__(self, image_size=448, annotation_file="archive/ctw1500_train_labels", img_folder="archive/train_images"):
        self.fp = pathlib.Path(annotation_file);
        self.img_folder = pathlib.Path(img_folder);
        self.image_size = image_size;
        self.annotations = {};

        for xml_file in self.fp.iterdir():
            root = ET.parse(xml_file).getroot();
            for image in root:
                # file name as id
                id = image.attrib['file'];
                self.annotations[id] = {};
                for i, box in enumerate(image):
                    h, w = int(box.attrib['height']), int(box.attrib['width'])
                    x1, y1 = int(box.attrib['left']), int(box.attrib['top']);
                    # [x1,y1, x2, y2] format
                    self.annotations[id]['box{}'.format(i)] = [x1, y1, x1+w, y1+h];
    
    def __len__(self):
        return len(self.annotations);

    def __getitem__(self, idx):
        img_key = list(self.annotations.keys())[idx];
        img_info = self.annotations[img_key];
        img_path = self.img_folder / img_key;
        img = cv.imread(str(img_path.absolute()));
        height, width = img.shape[:2];
        
        target = np.array([0,0,0,0,-1], ndmin=2).astype(np.float32);
        # max number of detections in the dataset is 66
        target = np.repeat(target, 66, axis=0);
        for i, key in enumerate(img_info):
            target[i][:4] = np.array(img_info[key]);
            target[i][4] = 1;
        
        target[:, 0::2][:,:-1] = target[:, 0::2][:,:-1] * float(self.image_size) / float(width);
        target[:, 1::2] *= float(self.image_size) / float(height);

        img = cv.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv.INTER_LINEAR);
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB);

        # convert numpy ndarray to tensor
        img = img.transpose((2,0,1)).astype(np.float32);
        return torch.from_numpy(img), torch.from_numpy(target);




class CocoDataset2014(Dataset):
    def __init__(self,image_size=448, annotation_file="cocotext.v2.json", img_folder="train2014", device="cpu") -> None:
        self.device = device;
        self.image_size = image_size;
        self.fp = pathlib.Path(annotation_file);
        self.img_folder = pathlib.Path(img_folder);
        with self.fp.open() as file:
            self.annotations = json.load(file);

        self.train_imgs = [];
        self.test_imgs = [];
        self.val_imgs = [];
        # get only the train images
        for img_id in self.annotations["imgs"].keys():
            img = self.annotations["imgs"][img_id];
            # check if the img is even in the img_folder
            img_file_path = self.img_folder / img["file_name"];
            if(not img_file_path.exists()):
                print(f"Could not find {img['file_name']} in {self.img_folder}");
                continue;
            # for image with no annotations that means no text associated with that image
            img["annotation_ids"] = self.annotations['imgToAnns'][img_id]
            if(img["set"] == "train"):
                self.train_imgs.append(img);
            elif(img["set"] == "test"):
                self.test_imgs.append(img);
            elif(img["set"] == "val"):
                self.val_imgs.append(img);
    def __len__(self):
        return  len(self.train_imgs);
    # output: img, target
    # target: [x1, y1, x2, y2, cls]
    def __getitem__(self, idx):
        img_info = self.train_imgs[idx]; # dictionary item
        img_path = self.img_folder / img_info["file_name"];
        img = cv.imread(str(img_path.absolute()));
        height, width = img.shape[:2];
        
        # -1 symbolizes no object class
        target = np.array([0,0,0,0,-1], ndmin=2).astype(np.float32);
        # the max number of bounding boxes in one image is 208
        target = np.repeat(target, 208, axis=0);
        if(len(img_info["annotation_ids"])):
            for j, id in enumerate(img_info["annotation_ids"]):
                target[j, :4] = self.annotations["anns"][str(id)]["bbox"];
                target[j, 4] = 1;
            # change [x1, y1, w, h] --> [x1, y1, x2, y2]
            target[:, 2] = target[:, 0] + target[:, 2];
            target[:, 3] = target[:, 1] + target[:, 3];
        
        target[:, 0::2][:,:-1] = target[:, 0::2][:,:-1] * float(self.image_size) / float(width);
        target[:, 1::2] *= float(self.image_size) / float(height);

        img = cv.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv.INTER_LINEAR);
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB);

        # convert numpy ndarray to tensor
        img = img.transpose((2,0,1)).astype(np.float32);
        return torch.from_numpy(img), torch.from_numpy(target);
            
