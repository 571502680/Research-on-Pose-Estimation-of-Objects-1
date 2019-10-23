import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from detect.eval.src.utils import IoU, transform_coord


class MaxPool1s(nn.Module):
    """Max pooling layer with stride 1"""

    def __init__(self, kernel_size):
        super(MaxPool1s, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    """Empty layer for shortcut connection"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """Detection layer

    Args
    - anchors: (list) list of anchor box sizes tuple
    - num_classes: (int) # classes
    - reso: (int) original image resolution
    - ignore_thresh: (float)
    """

    def __init__(self, anchors, num_classes, reso, ignore_thresh):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.reso = reso
        self.ignore_thresh = ignore_thresh

    def forward(self, x, y_true=None):
        """
        Transform feature map into 2-D tensor. Transformation includes
        1. Re-organize tensor to make each row correspond to a bbox
        2. Transform center coordinates
          bx = sigmoid(tx) + cx
          by = sigmoid(ty) + cy
        3. Transform width and height
          bw = pw * exp(tw)
          bh = ph * exp(th)
        4. Activation

        Args
        - x: (Tensor) feature map with size [bs, (5+nC)*nA, gs, gs]
            5 => [4 offsets (xc, yc, w, h), objectness]

        Returns
        - detections: (Tensor) feature map with size [bs, nA, gs, gs, 5+nC]
        """
        bs, _, gs, _ = x.size()
        stride = self.reso // gs
        num_attrs = 5 + self.num_classes
        nA = len(self.anchors)
        scaled_anchors = torch.Tensor([(a_w / stride, a_h / stride)
                                       for a_w, a_h in self.anchors]).cuda()
        grid_x = torch.arange(gs).repeat(gs, 1).view(
            [1, 1, gs, gs]).float().cuda()
        grid_y = torch.arange(gs).repeat(gs, 1).t().view(
            [1, 1, gs, gs]).float().cuda()
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Re-organize [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
        x = x.view(bs, nA, num_attrs, gs, gs).permute(
            0, 1, 3, 4, 2).contiguous()
        detections = torch.Tensor(bs, nA, gs, gs, num_attrs).cuda()
        detections[..., 0] = torch.sigmoid(x[..., 0]) + grid_x
        detections[..., 1] = torch.sigmoid(x[..., 1]) + grid_y
        detections[..., 2] = torch.exp(x[..., 2]) * anchor_w
        detections[..., 3] = torch.exp(x[..., 3]) * anchor_h
        detections[..., :4] *= stride
        detections[..., 4] = torch.sigmoid(x[..., 4])
        detections[..., 5:] = torch.sigmoid(x[..., 5:])

        return detections.view(bs, -1, num_attrs)


class NMSLayer(nn.Module):
    """
    NMS layer which performs Non-maximum Suppression
    1. Filter background
    2. Get detection with particular class
    3. Sort by confidence
    4. Suppress non-max detection
    """

    def __init__(self, conf_thresh, iou_thresh):
        """
        Args
        - conf_thresh: (float) fore-ground confidence threshold
        - iou_thresh: (float) iou threshold
        """
        super(NMSLayer, self).__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def forward(self, x):
        """
        Args
        - x: (Tensor) detection feature map, with size [bs, num_bboxes, [x,y,w,h,p_obj]+num_classes]

        Returns
        - detections: (Tensor) detection result with size [num_bboxes, [image_batch_idx, 4 offsets, p_obj, max_conf, cls_idx]]
        """
        bs, num_bboxes, num_attrs = x.size()
        detections = torch.Tensor().cuda()

        for idx in range(bs):
            pred = x[idx]

            try:
                non_zero_pred = pred[pred[:, 4] > self.conf_thresh]
                non_zero_pred[:, :4] = transform_coord(
                    non_zero_pred[:, :4], src='center', dst='corner')
                max_score, max_idx = torch.max(non_zero_pred[:, 5:], 1)
                max_idx = max_idx.float().unsqueeze(1)
                max_score = max_score.float().unsqueeze(1)
                non_zero_pred = torch.cat(
                    (non_zero_pred[:, :5], max_score, max_idx), 1)
                classes = torch.unique(non_zero_pred[:, -1])
            except Exception:  # no object detected
                continue

            for cls in classes:
                cls_pred = non_zero_pred[non_zero_pred[:, -1] == cls]
                conf_sort_idx = torch.sort(cls_pred[:, 4], descending=True)[1]
                cls_pred = cls_pred[conf_sort_idx]
                max_preds = []
                while cls_pred.size(0) > 0:
                    max_preds.append(cls_pred[0].unsqueeze(0))
                    ious = IoU(max_preds[-1], cls_pred)
                    cls_pred = cls_pred[ious < self.iou_thresh]

                if len(max_preds) > 0:
                    max_preds = torch.cat(max_preds).data
                    batch_idx = max_preds.new(max_preds.size(0), 1).fill_(idx)
                    seq = (batch_idx, max_preds)
                    detections = torch.cat(seq, 1) if detections.size(
                        0) == 0 else torch.cat((detections, torch.cat(seq, 1)))

        return detections
