import os
import sys
import cv2
import torch
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms

from data.linemod.sixd import SixdToolkit
from detect.eval.src.detector import Detector
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.config import prepare_weight, prepare_cfg
from utils import draw_keypoints, crop_from_dets, draw_heatmap
sys.path.append('./keypoint/train_sppe')
from keypoint.train_sppe.main_fast_inference import InferenNet_fast
from keypoint.train_sppe.utils.eval import getPrediction
from keypoint.train_sppe.utils.img import im_to_torch


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    parser.add_argument('--bs', default=1,type=int, help="Batch size")
    parser.add_argument('--reso', default=416,type=int, help="Image resolution")
    parser.add_argument('--gpu', default='0,1,2,3', help="GPU ids")
    parser.add_argument('--name', default='linemod-single',type=str, choices=['linemod-single', 'linemod-occ'])
    parser.add_argument('--seq', default='01',type=str, help="Sequence number")
    parser.add_argument('--ckpt', type=str, help="Checkpoint path")
    return parser.parse_args()


args = parse_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='cluster', is_train=False)
    _, val_dataloder = prepare_dataset(args.name, args.reso, args.bs, args.seq)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt)
    )
    pose_model = InferenNet_fast(
        dataset=args.name,
        kernel_size=5,
        seqname=args.seq,
        kpnum=17,
        kptype='cluster'
    )
    pose_model = pose_model.cuda()

    tbar = tqdm(val_dataloder)
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        img_path = meta['path'][0]
        idx = img_path.split('/')[-1].split('.')[0]

        inputs = inputs.cuda()
        with torch.no_grad():
            try:
                bboxes, confs = detector.detect(inputs)
            except Exception:
                # No object found
                continue

            orig_img = cv2.imread(meta['path'][0])
            orig_inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            cropped_inputs, pt1, pt2 = crop_from_dets(
                orig_inp, bboxes[0], 320, 256)
            hms = pose_model(cropped_inputs.unsqueeze(0).cuda()).cpu()

            try:
                _, pred_kps, pred_kps_score = getPrediction(
                hms, pt1.unsqueeze(0), pt2.unsqueeze(0), 320, 256, 80, 64)
            except Exception:
                continue

        if args.name == 'linemod-single':
            f = bench.frames[args.seq][int(idx)]
        elif args.name == 'linemod-occ':
            f = bench.frames['02'][int(idx)]
        annot = f['annots'][f['obj_ids'].index(int(args.seq))]
        gt_kps = annot['kps']

        # save_dir = os.path.join('./results/hms/%s' % idx)
        # draw_heatmap(hms[0], save_dir)

        save_path = os.path.join('./results/kps/%s.png' % idx)
        draw_keypoints(img_path, gt_kps, pred_kps[0].numpy(), bboxes[0].numpy(),
                       pred_kps_score[0].squeeze().numpy(), save_path)
