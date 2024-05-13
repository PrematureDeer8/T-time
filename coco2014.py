import pathlib
import json
import cv2 as cv
import torch

class CocoDataset2014:
    def __init__(self, annotation_file, img_folder, shuffle=False) -> None:
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
    # input:
    #   images (list of tensors) -> tensor: shape [batch_size, channel, height, width]
    #   targets -> [{"boxes": [[1,2,3,4],[5,6,7,8]], "labels": [10, 13] }, ...]
    def input_loader(self, batch_size, height=640, width=640):
        img_batch = [];
        targets = [];
        img_batch = torch.zeros(batch_size, 3, height, width);
        # the max number of bounding boxes in one image is 208 
        bbox_batch = torch.zeros(batch_size, 208, 4);
        labels = torch.zeros(batch_size, 208);
        for i in range(batch_size):
            file = self.img_folder / self.train_imgs[i]["file_name"];
            img_np = cv.imread(str(file.absolute()));
            ogh, ogw = img_np.shape[:2];
            img_np = cv.resize(img_np, dsize=(height, width), interpolation=cv.INTER_LINEAR);
            # change numpy dim of [3,height, width ] --> [width, height,3]
            img = torch.from_numpy(img_np).permute(2,1,0);
            img_batch[i] = img;
            d = {};
            for j, box_id in enumerate(self.train_imgs[i]["annotation_ids"]):
                # should be a list [x1, y1, width, height]
                bbox = self.annotations["anns"][str(box_id)]["bbox"];
                if(len(bbox)):
                    # have to change the bounding box to match
                    # the resized image
                    scale_w, scale_h = width/ogw, height/ogh;
                    x1, y1, x2, y2 = bbox[0] * scale_w, bbox[1] * scale_h,\
                        (bbox[0] + bbox[2]) * scale_w, (bbox[1] + bbox[3]) * scale_h;
                    bbox_batch[i][j] = torch.Tensor([x1,y1,x2,y2]);
                    # change the label to be an object in respective position
                    labels[i][j] = 1;
            d["boxes"] = bbox_batch[i];
            d["labels"] = labels[i];
            targets.append(d);
        return img_batch, targets;