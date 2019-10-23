import os
import torch
import random
import datetime
import numpy as np
from PIL import Image, ImageFont, ImageDraw
opj = os.path.join


def parse_cfg(cfgfile):
    """Parse a configuration file

    Args
    - cfgfile: (str) path to config file

    Returns
    - blocks: (list) list of blocks, with each block describes a block in the NN to be built
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # skip empty lines
    lines = [x for x in lines if x[0] != '#']  # skip comment
    lines = [x.rstrip().lstrip() for x in lines]
    file.close()

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    blocks = blocks[1:]

    return blocks


def parse_detection(detection, reso, cls_idx=None):
    """
    Parse detection result

    Args
    - detection: (np.array) Detection result for one image
        [#bbox, [batch_idx, x1, y1, x2, y2, objectness, conf, class idx]]
    - reso: (int) Image resolution

    Returns
    - area: (Tensor) With size [4,]
    - conf: (float) Confidence score
    """
    # FIXME: constants
    h_ratio = 480 / reso
    w_ratio = 640 / reso

    if cls_idx is not None:
        detection = detection[detection[:, -1] == cls_idx]

    if detection.size(0) == 0:
        return None, None
    else:
        best_idx = np.argmax(detection[:, -3])
        bbox = detection[best_idx, 1:5]
        conf = float(detection[best_idx, 6])

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    area = torch.Tensor((x1 * w_ratio, y1 * h_ratio,
                         x2 * w_ratio, y2 * h_ratio))

    return area, conf


def transform_coord(bbox, src='center', dst='corner'):
    """Transform bbox coordinates
      |---------|           (x1,y1) *---------|
      |         |                   |         |
      |  (x,y)  h                   |         |
      |         |                   |         |
      |____w____|                   |_________* (x2,y2)
         center                         corner

    @Args
      bbox: (Tensor) bbox with size [..., 4]

    @Returns
      bbox_transformed: (Tensor) bbox with size [..., 4]
    """
    flag = False
    if len(bbox.size()) == 1:
        bbox = bbox.unsqueeze(0)
        flag = True

    bbox_transformed = bbox.new(bbox.size())
    if src == 'center' and dst == 'corner':
        bbox_transformed[..., 0] = (bbox[..., 0] - bbox[..., 2]/2)
        bbox_transformed[..., 1] = (bbox[..., 1] - bbox[..., 3]/2)
        bbox_transformed[..., 2] = (bbox[..., 0] + bbox[..., 2]/2)
        bbox_transformed[..., 3] = (bbox[..., 1] + bbox[..., 3]/2)
    elif src == 'corner' and dst == 'center':
        bbox_transformed[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
        bbox_transformed[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
        bbox_transformed[..., 2] = bbox[..., 2] - bbox[..., 0]
        bbox_transformed[..., 3] = bbox[..., 3] + bbox[..., 1]

    if flag == True:
        bbox_transformed = bbox_transformed.squeeze(0)

    return bbox_transformed


def IoU(box1, box2, format='corner'):
    """Compute IoU between box1 and box2

    Args
    - box: (torch.cuda.Tensor) bboxes with size [# bboxes, 4]  # TODO: cpu
    - format: (str) bbox format
        'corner' => [x1, y1, x2, y2]
        'center' => [xc, yc, w, h]
    """
    if format == 'center':
        box1 = transform_coord(box1)
        box2 = transform_coord(box2)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * \
        torch.max(inter_rect_y2 - inter_rect_y1 + 1,
                  torch.zeros(inter_rect_x2.shape).cuda())
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def crop_img(img_path, detection, reso, cls_idx=None):
    """Crop target object in image

    Args
    - img_path: (str) path to one image
    - detection: (np.array) detection result for one image
        [#bbox, [batch_idx, top-left x, top-left y, bottom-right x, bottom-right y, objectness, conf, class idx]]
    - reso: (int) image resolution

    Returns
    - area: (tuple) bbox
    """
    img = Image.open(img_path)
    w, h = img.size
    h_ratio = h / reso
    w_ratio = w / reso

    if cls_idx is not None:
        detection = detection[detection[:, -1] == cls_idx]

    if detection.size(0) == 0:
        return None
    else:
        best_idx = np.argmax(detection[:, -3])
        bbox = detection[best_idx, 1:5]

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    area = torch.Tensor((x1 * w_ratio, y1 * h_ratio,
                         x2 * w_ratio, y2 * h_ratio))

    return area


def crop_img_all(img_path, detection, reso):
    img = Image.open(img_path)
    w, h = img.size
    h_ratio = h / reso
    w_ratio = w / reso

    areas = []

    for det in detection:
        bbox = det[1:5]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        objectness = 'conf: %.2f' % (det[-3] * 100)
        cls_score = 'cls: %.2f' % (det[-2] * 100)
        area = (x1 * w_ratio, y1 * h_ratio, x2 * w_ratio,
                y2 * h_ratio, objectness, cls_score)
        areas.append(area)

    return areas
