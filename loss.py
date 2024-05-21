from torch import nn
import torch
from torchvision.ops import box_iou

# check device
if(torch.cuda.is_available()):
    # torch.set_default_device("cuda");
    device = ("cuda");
elif(torch.backends.mps.is_available()):
    # torch.set_default_device(device);
    device = ("mps");
else:
    device = ("cpu");
class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        noobj_scaling_factor = 1e-2;
        obj_scaling_factor = 1e2;
        alpha = 0.25;
        gamma = 2.0;
        batch_size = classifications.shape[0];
        classification_losses = [];
        regression_losses = [];
        
        anchor = anchors[0];

        anchor_widths = anchor[:, 2] - anchor[:, 0];
        anchor_heights = anchor[:, 3] - anchor[:, 1];
        anchor_ctr_x = anchor_widths * 0.5 + anchor[:, 0];
        anchor_ctr_y = anchor_heights * 0.5 + anchor[:, 1];

        for j in range(batch_size):
            classification = classifications[j].to(device);
            regression = regressions[j].to(device);

            #annotations: [x1, y1, x2, y2, cls]
            # class of -1 means no object 
            bbox_annotation = annotations[j].to(device);
            # get all the bounding box annotations that are of object
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1];

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4);
            #  jth image form batch has no object 
            if(bbox_annotation.shape[0] == 0):
                alpha_factor = torch.ones(classification.shape) * alpha;

                alpha_factor = 1. - alpha_factor;
                focal_weight = classification;
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma);

                bce = -(torch.log(1.0 - classification));

                cls_loss = focal_weight * bce;
                classification_losses.append(cls_loss.sum());
                regression_losses.append(torch.tensor(0).float());
                continue;
            
            IoU = box_iou(anchor, bbox_annotation[..., :4]); # num_anchors x num_annotations
            # takes the max IoU each anchor has with a groundtruth
            IoU_max, IoU_argmax = torch.max(IoU, dim=1); # num_anchors x 1

            targets = torch.ones(classification.shape).to(device) * -1;
            #assign zero to IoU less than 0.4
            targets[torch.lt(IoU_max, 0.4), : ] = 0;
            positive_indices = torch.ge(IoU_max, 0.5).to(device);

            num_positive_anchors = positive_indices.sum();
            # each anchor gets assigned a groundtruth
            # where the corresponding ground truth is the max IoU with the anchor
            assigned_annotations = bbox_annotation[IoU_argmax, :];

            # find all anchors greater than 0.5 IoU 
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            elif torch.backends.mps.is_available():
                alpha_factor = torch.ones(targets.shape).to(device) * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.).to(device), alpha_factor, 1. - alpha_factor)
            # focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            # focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * obj_scaling_factor * torch.log(classification).to(device) + (1.0 - targets) * noobj_scaling_factor * torch.log(1.0 - classification).to(device))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = alpha_factor * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            elif torch.backends.mps.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0).to(device))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                elif torch.backends.mps.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device);
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                elif(torch.backends.mps.is_available()):
                    regression_losses.append(torch.tensor(0).float().to(device))
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)