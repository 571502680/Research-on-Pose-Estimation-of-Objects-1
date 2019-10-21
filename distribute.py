import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from data.linemod.sixd import SixdToolkit
from detect.eval.src.detector import Detector
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.config import prepare_weight, prepare_cfg


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--reso', type=int, help="Image resolution")
    parser.add_argument('--gpu', default='0,1,2,3', help="GPU ids")
    parser.add_argument('--name', type=str, choices=['linemod-single'])
    parser.add_argument('--seq', type=str, help="Sequence number")
    return parser.parse_args()


args = parse_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='sift', is_train=False)
    _, val_dataloder = prepare_dataset(args.name, args.reso, args.bs, args.seq)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.seq + '.best.weights')
    )

    xmin, xmax, ymin, ymax = [], [], [], []

    tbar = tqdm(val_dataloder)
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        img_path = meta['path'][0]
        idx = img_path.split('/')[-1].split('.')[0]

        inputs = inputs.cuda()
        with torch.no_grad():
            bboxes, confs = detector.detect(inputs)

        pred_bbox = bboxes[0].numpy()

        f = bench.frames[args.seq][int(idx)]
        annot = f['annots'][f['obj_ids'].index(int(args.seq))]
        gt_bbox = annot['bbox']

        xmin.append((gt_bbox[0] - pred_bbox[0]) / 640.)
        ymin.append((gt_bbox[1] - pred_bbox[1]) / 480.)
        xmax.append((gt_bbox[2] - pred_bbox[2]) / 640.)
        ymax.append((gt_bbox[3] - pred_bbox[3]) / 480.)

    xmin = np.array(xmin)
    xmax = np.array(xmax)
    ymin = np.array(ymin)
    ymax = np.array(ymax)

    dist = np.array([
        xmin.mean(), xmin.std(), xmax.mean(), xmax.std(),
        ymin.mean(), ymin.std(), ymax.mean(), ymax.std()
    ])

    os.makedirs('./results/distribution/%s' % args.seq, exist_ok=True)
    np.save('./results/distribution/%s/xmin.npy' % args.seq, xmin)
    np.save('./results/distribution/%s/xmax.npy' % args.seq, xmax)
    np.save('./results/distribution/%s/ymin.npy' % args.seq, ymin)
    np.save('./results/distribution/%s/ymax.npy' % args.seq, ymax)
    np.save('./results/distribution/%s/dist.npy' % args.seq, dist)
