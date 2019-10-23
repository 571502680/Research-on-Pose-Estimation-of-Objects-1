import os
import torch
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .config import class_name, class_idx
from detect.eval.src.model import YOLOv3
from detect.eval.src.utils import crop_img, parse_detection, IoU
from detect.eval.src.layers import NMSLayer
opj = os.path.join


class Detector:
    def __init__(self, cfgfile, seq, weightfile, reso=416, iou_thresh=0.5, conf_thresh=0.01):
        """
        Args
        - cfgfile: (str) Configuration file
        - seq: (str) Sequence number
        - reso: (int) Image resolution
        - iou_thresh: (float) IoU threshold for NMS
        - conf_thresh: (float) Objectness score threshold for NMS
        """
        self.yolo = YOLOv3(cfgfile, reso).cuda()
        self.yolo.eval()
        self.nms = NMSLayer(conf_thresh, iou_thresh).cuda()
        self.yolo.load_weights(weightfile)

        self.seq = seq
        self.dataset = cfgfile.split('/')[-1].split('.')[0]
        self.seqname = class_name(self.dataset, self.seq)
        if self.dataset == 'linemod-single':
            self.cls_idx = 0
        else:
            self.cls_idx = class_idx(self.dataset, self.seqname)

    def detect(self, inputs):
        """
        Args
        - inputs: (Tensor) With size [BS x C x W x H]

        Returns
        - bboxes: (Tensor) With size [BS x 4]
        - confs: (Tensor) With size [BS x 1]
        """
        bs = inputs.size(0)
        bboxes = torch.Tensor(bs, 4)
        confs = torch.Tensor(bs, 1)
        detections = self.nms(self.yolo(inputs))
        for idx in range(bs):
            detection = detections[detections[:, 0] == idx]
            bbox, conf = parse_detection(detection.cpu(), self.yolo.reso, cls_idx=self.cls_idx)
            bboxes[idx] = bbox
            confs[idx] = conf
            if bbox is None:
                raise Exception

        return bboxes, confs

    def crop(self, inputs, bboxes, w1, h1, w2, h2):
        """
        Args
        - inputs: (Tensor) With size [BS x C x W x H]
        - bboxes: (Tensor) With size [BS x 4]
        - w1: (int) Original image's width
        - h1: (int) Original image's height
        - w2: (int) Scale cropped inputs' width
        - h2: (int) Scale cropped inputs' height

        Returns
        - outputs: (Tensor) With size [BS x C x width x h

        eight]
        """
        assert inputs.size(0) == bboxes.size(0)
        bs, c, reso, _ = inputs.size()
        outputs = torch.Tensor(bs, c, w2, h2)
        for idx in range(inputs.size(0)):
            x1, y1, x2, y2 = bboxes[idx, :4]
            x1, x2 = int(x1 * reso / w1), int(x2 * reso / w2)
            y1, y2 = int(y1 * reso / h1), int(y2 * reso / h2)
            cropped_inputs = inputs[idx, :, x1:x2, y1:y2].unsqueeze(0)
            outputs[idx] = F.interpolate(cropped_inputs, size=(w2, h2),
                                         mode='bilinear', align_corners=True)
        return outputs

    def detect_all(self, dataloader, savedir=None):
        """
        Iterate dataloader and detect every object in it.

        Args:
        - dataloader: (Dataloader)
        - savedir: (str, optional)
        """
        bboxes_list = []
        tbar = tqdm(dataloader)
        for batch_idx, (inputs, labels, meta) in enumerate(tbar):
            batch_size = inputs.size(0)
            inputs = inputs.cuda()
            detections = self.nms(self.yolo(inputs))

            for idx in range(batch_size):
                img_path = meta['path'][idx]
                img_name = img_path.split('/')[-1]
                img_cls = img_path.split('/')[-3]
                label = labels[idx]
                if list(detections.cpu().numpy())==[]:
                    continue
                print(detections)
                print(detections[:, 0])
                detection = detections[detections[:, 0] == idx]
                bbox = crop_img(img_path, detection.cpu(), self.yolo.reso, cls_idx=self.cls_idx)
                if savedir is not None:
                    img = Image.open(img_path)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]))
                    img.save(opj(savedir, img_name))
