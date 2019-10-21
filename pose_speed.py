import os
import sys
import cv2
import time
import pickle
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel

from utils import *
from data.linemod.sixd import SixdToolkit
from detect.eval.src.detector import Detector
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.config import prepare_cfg, prepare_weight
sys.path.append('./keypoint/train_sppe')
from keypoint.train_sppe.utils.img import im_to_torch
from keypoint.train_sppe.utils.eval import getPrediction
from keypoint.train_sppe.main_fast_inference import InferenNet_fast


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    parser.add_argument('--reso', type=int, help="Image resolution")
    parser.add_argument('--kptype', type=str, help="Keypoint type")
    parser.add_argument('--kpnum', type=int, help="Checkpoint path")
    parser.add_argument('--gpu', default='0,1,2,3', help="GPU ids")
    parser.add_argument('--name', type=str,
                        choices=['linemod-single', 'linemod-occ'])
    parser.add_argument('--seq', type=str, help="Sequence number")
    parser.add_argument('--ckpt', type=str, help="Checkpoint path")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('-save', action='store_true', help="Save pose figure")
    return parser.parse_args()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def val(model, detector, dataloader, bench, seq):
    device = torch.device("cuda")
    tbar = tqdm(dataloader)
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        img_path = meta['path'][0]
        idx = img_path.split('/')[-1].split('.')[0]
        inputs = inputs.to(device)
        with torch.no_grad():
            # object detection
            try:
                bboxes, confs = detector.detect(inputs)
            except Exception:
                # No object found
                print("No object found")
                continue

            # keypoint localization
            orig_img = cv2.imread(meta['path'][0])
            orig_inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            cropped_inputs, pt1, pt2 = crop_from_dets(
                orig_inp, bboxes[0], 320, 256)
            hms = model(cropped_inputs.unsqueeze(0).cuda()).cpu()
            try:
                _, pred_kps, pred_kps_score = getPrediction(
                    hms, pt1.unsqueeze(0), pt2.unsqueeze(0), 320, 256, 80, 64)
            except Exception:
                print("Jump Error frame", idx)
                continue

            # pose estimation
            K = 17
            best_idx = np.argsort(pred_kps_score[0, :, 0]).flip(0)
            best_k = best_idx[:K]

            pred_pose = bench.solve_pnp(
                bench.kps[seq][best_k], pred_kps[0][best_k].numpy())

            # result[int(idx)] = {
            #     'bbox': bboxes[0].numpy(),
            #     'kps': pred_kps[0].numpy(),
            #     'pose': pred_pose
            # }



if __name__ == '__main__':
    args = parse_arg()
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=args.kpnum,
                        kptype=args.kptype, is_train=False)
    kp3d = bench.models[args.seq]

    num_gpus = len(args.gpu.split(','))
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    _, val_dataloder = prepare_dataset(args.name, args.reso, 1, args.seq, distributed=True)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt)
    )
    pose_model = InferenNet_fast(
        dataset=args.name,
        kernel_size=5,
        seqname=args.seq,
        kpnum=args.kpnum,
        kptype=args.kptype
    ).cuda()
    detector.yolo = DistributedDataParallel(detector.yolo)
    detector.nms = DistributedDataParallel(detector.nms)
    pose_model = DistributedDataParallel(pose_model)

    start_time = time.time()
    val(pose_model, detector, val_dataloder, bench, args.seq)
    elapsed = time.time() - start_time
    from IPython import embed
    embed()

    print("[LOG] Use", elapsed, "seconds")
    print("[LOG] Done!")
