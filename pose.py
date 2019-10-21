#*/****************************************evalution******************************************/*
import os
import sys
import cv2
import pickle
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms

from utils import *
from data.linemod.sixd import SixdToolkit
from detect.eval.src.detector import Detector
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.config import prepare_cfg, prepare_weight
sys.path.append('/home/liqi/PycharmProjects/kp6d/keypoint/home/liqi/PycharmProjects/kp6d/keypoint'
                '/keypoint/train_sppe')
from keypoint.train_sppe.main_fast_inference import InferenNet_fast
from keypoint.train_sppe.utils.eval import getPrediction
from keypoint.train_sppe.utils.img import im_to_torch


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    # parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--reso', default='416',type=int, help="Image resolution")
    parser.add_argument('--kptype', default='cluster',type=str, help="Keypoint type")
    parser.add_argument('--kpnum', default=17,type=int, help="Checkpoint path")
    parser.add_argument('--topk', default=9,type=int, help="Checkpoint path")
    parser.add_argument('--gpu', default='1,2,3', help="GPU ids")
    parser.add_argument('--name', default='linemod-single',type=str, choices=['linemod-single', 'linemod-occ'])
    parser.add_argument('--seq', default='01',type=str, help="Sequence number")
    parser.add_argument('--ckpt',default='/home/common/liqi/data/LINEMOD_6D/LM6d_origin/data_linemod_gt/02/yolo-linemod-single_6200.weights', type=str, help="Checkpoint path")
    parser.add_argument('-save', default=True, help="Save pose figure")
    return parser.parse_args()


args = parse_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=args.kpnum,
                        kptype=args.kptype, is_train=False)
    kp3d = bench.models[args.seq]
    pose_ckpt = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/exp/final_model-%s-%s-%s/%s.pkl' % (
        args.name, args.kpnum, args.kptype, args.seq)

    _, val_dataloder = prepare_dataset(args.name, args.reso, 1, args.seq)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt)
    )
    pose_model = InferenNet_fast(
        kpnum=args.kpnum,
        path=pose_ckpt
    ).cuda()

    tbar = tqdm(val_dataloder)
    result = dict()
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        img_path = meta['path'][0]
        idx = img_path.split('/')[-1].split('.')[0]
        inputs = inputs.cuda()
        with torch.no_grad():
            # object detection
            try:
                bboxes, confs = detector.detect(inputs)#object detection supply bboxes, confs.
            except Exception:
                # No object found
                # print("detection failed")
                continue

            # keypoint localization
            orig_img = cv2.imread(meta['path'][0])
            orig_inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            cropped_inputs, pt1, pt2 = crop_from_dets(
                orig_inp, bboxes[0], 320, 256)#crop feature map to adopt the crop picture!
            # cropped_inputs=cropped_inputs.copy().permute(1,2,0)
            # import matplotlib.pyplot as plt
            # plt.imshow(cropped_inputs)
            # plt.show()
            hms = pose_model(cropped_inputs.unsqueeze(0).cuda()).cpu()
            try:
                _, pred_kps, pred_kps_score = getPrediction(
                    hms, pt1.unsqueeze(0), pt2.unsqueeze(0), 320, 256, 80, 64)
            except Exception:
                # print("Jump Error frame", idx)
                continue

            # pose estimation
            K = args.topk
            best_idx = np.argsort(pred_kps_score[0, :, 0]).flip(0)
            best_k = best_idx[:K]

            pred_pose = bench.solve_pnp(
                bench.kps[args.seq][best_k], pred_kps[0][best_k].numpy())

            result[int(idx)] = {
                'bbox': bboxes[0].numpy(),
                'kps': pred_kps[0].numpy(),
                'pose': pred_pose
            }

            # if args.save is True:
            #     f = bench.frames[args.seq][int(idx)]
            #     annot = f['annots'][f['obj_ids'].index(int(args.seq))]
            #     gt_pose = annot['pose']
            #
            #     save_path = os.path.join('/home/liqi/PycharmProjects/kp6d/results/pose/%s.png' % idx)
            #     draw_6d_pose(img_path, gt_pose, pred_pose,
            #                  kp3d, bench.cam, save_path)

    with open('/home/liqi/PycharmProjects/kp6d/results/%s.pkl' % args.seq, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("[LOG] Done!")
