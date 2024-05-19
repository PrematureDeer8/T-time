from torchvision.ops import nms # type: ignore
import torch # type: ignore

def anchor_nms(classification, boxes, deltas, std=torch.Tensor([0.1, 0.1, 0.2, 0.2]), mean=torch.Tensor([0,0,0,0])):

    widths = boxes[..., 2] - boxes[..., 0];
    heights = boxes[..., 3] - boxes[..., 1];
    ctr_x = boxes[..., 0] + 0.5 * widths;
    ctr_y = boxes[..., 1] + 0.5 * heights;

    dx = deltas[..., 0] * std[0] + mean[0];
    dy = deltas[..., 1] * std[1] + mean[1];
    dw = deltas[..., 2] * std[2] + mean[2];
    dh = deltas[..., 3] * std[3] + mean[3];

    # add offsets from regression subnet to anchor boxes
    pred_ctr_x = ctr_x + dx * widths;
    pred_ctr_y = ctr_y + dy * heights;
    pred_w = torch.exp(dw) * widths;
    pred_h = torch.exp(dh) * heights;

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w;
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h;
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w;
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h;

    pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2);

    finalAnchorBoxesIndexes = torch.Tensor([]).long();
    finalAnchorBoxesCoordinates = torch.Tensor([]);
    finalScores = torch.Tensor([]);

    for i in range(classification.shape[2]):
        scores = torch.squeeze(classification[..., i]);
        scores_over_thresh = (scores > 0.05);
        if(scores_over_thresh.sum() == 0):
            continue;

        scores = scores[scores_over_thresh];
        anchorBoxes = torch.squeeze(pred_boxes);
        anchorBoxes = anchorBoxes[scores_over_thresh];
        anchors_nms_idx = nms(anchorBoxes, scores, 0.5);

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]));
        finalAnchorBoxesIndexesValue  = torch.Tensor([i] * anchors_nms_idx.shape[0]);

        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue));
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]));

    
    return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates];
