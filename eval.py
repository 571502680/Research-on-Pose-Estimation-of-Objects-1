import pickle
import argparse
from tqdm import tqdm

from metrics import *
from data.linemod.sixd import SixdToolkit

NAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup',
         'driller', 'duck', 'eggbo', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')


def parse_arg():
    parser = argparse.ArgumentParser(description='KP6D evaluation')
    parser.add_argument('--dataset',default='linemod-single', type=str, help="Dataset name")
    parser.add_argument('--seq',default='01', type=str, help="Sequence number")
    return parser.parse_args()


args = parse_arg()

if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='cluster', is_train=False)
    kp3d = bench.models[args.seq]
    diameter = bench.models_info[args.seq]['diameter']

    if args.dataset == 'linemod-single':
        frame_seq = args.seq
    elif args.dataset == 'linemod-occ':
        frame_seq = '02'

    with open('results/%s.pkl' % args.seq, 'rb') as handle:
        result = pickle.load(handle)

    add_errs = []
    adds = []
    proj_2d_errs = []
    ious = []
    for k, v in tqdm(result.items()):
        f = bench.frames[frame_seq][k]
        try:
            annot = f['annots'][f['obj_ids'].index(int(args.seq))]
        except:
            # object not found
            # print("object not found")
            continue
        gt_pose = annot['pose']
        gt_bbox = annot['bbox']

        pred_bbox = v['bbox']
        pred_pose = v['pose']

        iou = IoU(gt_bbox, pred_bbox)
        ious.append(iou)
        if iou > 0.5:
            # ADD
            if args.seq == '10' or args.seq == '11':
                add = ADDS_err(gt_pose, pred_pose, kp3d)
            else:
                add = ADD_err(gt_pose, pred_pose, kp3d)
            add_errs.append(add)
            adds.append(add < 0.1 * diameter)

            # 2D REPROJECTION ERROR
            err_2d = projection_error_2d(gt_pose, pred_pose, kp3d, bench.cam)
            proj_2d_errs.append(err_2d)
        else:
            pass
            # print("iou < 0.5")

    PIXEL_THRESH = 5
    mean_add_err = np.mean(add_errs)
    mean_add = np.mean(adds)
    mean_2d_err = np.mean(np.array(proj_2d_errs))
    mean_2d_acc = np.mean(np.array(proj_2d_errs) < PIXEL_THRESH)
    mean_iou = np.mean(np.array(ious) > 0.5)
    print("[LOG] Mean add error for seq %s %s is: %.3f" %
          (args.seq, NAMES[int(args.seq) - 1], mean_add_err))
    print("[LOG] Mean add accuracy for seq %s %s is: %.3f" %
          (args.seq, NAMES[int(args.seq) - 1], mean_add))
    print("[LOG] 2d reprojection error for seq %s %s is: %.3f" %
          (args.seq, NAMES[int(args.seq) - 1], mean_2d_err))
    print("[LOG] 2d reprojection accuracy for seq %s %s is: %.3f" %
          (args.seq, NAMES[int(args.seq) - 1], mean_2d_acc))
    print("[LOG] Mean IoU for seq %s %s is: %.3f" %
          (args.seq, NAMES[int(args.seq) - 1], mean_iou))
