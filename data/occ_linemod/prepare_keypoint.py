import os
import sys
import h5py
import argparse
import numpy as np
from tqdm import tqdm
opj = os.path.join

from sixd import SixdToolkit

LINEMOD = '/home/penggao/data/sixd/hinterstoisser/test/02'
KPDROOT = '/home/penggao/projects/pose/kp6d/keypoint/data/occ'
NAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup',
                'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
OCCNAMES = ('ape', 'can', 'cat', 'driller', 'duck',
            'eggbox', 'glue', 'holepuncher')


def parse_arg():
    parser = argparse.ArgumentParser(description='Synthetic data generator')
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--kpnum', choices=[17, 9],
                        type=int, help="Number of keypoints")
    parser.add_argument('--kptype', choices=['sift', 'cluster', 'corner'],
                        type=str, help="Type of keypoints")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    print("[LOG] Preparing h5 for KPD training")
    print("[LOG] Number of keypoints: %d" % args.kpnum)
    print("[LOG] Type of keypoints: %s" % args.kptype)
    print("[LOG] Sequence: %s %s" % (args.seq, NAMES[int(args.seq)-1]))

    if NAMES[int(args.seq) - 1] not in OCCNAMES:
        print("[ERROR] Sequence not exist in occlusion linemod")
        sys.exit()

    bench = SixdToolkit(dataset='hinterstoisser', kpnum=args.kpnum,
                        kptype=args.kptype, is_train=False)

    KPDROOT = opj(KPDROOT, str(args.kpnum), args.kptype, args.seq)

    if os.path.exists(KPDROOT):
        print("[WARNING] Overwriting existing images in %s" % KPDROOT)
        print("[WARNING] Proceed (y/[n])?", end=' ')
        choice = input()
        if choice == 'y':
            os.system('rm -r %s' % KPDROOT)
        else:
            print("[LOG] Terminated by user, exit")
            sys.exit()

    os.makedirs(KPDROOT)
    os.makedirs(opj(KPDROOT, 'train'))
    os.makedirs(opj(KPDROOT, 'eval'))

    print("[LOG] Preparing data")
    with open(opj(LINEMOD, 'train.txt'), 'r') as f:
        trainlists = f.readlines()
    trainlists = [x.strip() for x in trainlists]

    count = 0
    bboxes = {'train': [], 'eval': []}
    kps = {'train': [], 'eval': []}
    imgnames = {'train': [], 'eval': []}
    tbar = tqdm(bench.frames['02'])
    for idx, f in enumerate(tbar):
        tbar.set_description('%04d' % idx)
        tag = 'train' if '%04d' % idx in trainlists else 'eval'

        # annotation
        try:
            annot = f['annots'][f['obj_ids'].index(int(args.seq))]
        except:
            print("Sequence %s not exist in frame %d" % (args.seq, idx))
            continue
        count += 1
        name_chars = []
        for char in '%012d' % int(idx):
            name_chars.append(ord(char))
        bboxes[tag].append(annot['bbox'])
        kps[tag].append(annot['kps'])
        imgnames[tag].append(np.asarray(name_chars))

        # images
        srcimg = f['path']
        dstimg = opj(KPDROOT, tag, '%012d.png' % int(idx))
        os.symlink(srcimg, dstimg)

    for tag in ('train', 'eval'):
        with h5py.File(opj(KPDROOT, 'annot_%s.h5' % tag), "w") as f:
            f.create_dataset("imgname", data=np.vstack(imgnames[tag]))
            f.create_dataset("bndbox", data=np.vstack(
                bboxes[tag]).reshape(-1, 1, 4))
            f.create_dataset("part", data=np.vstack(
                kps[tag]).reshape(-1, args.kpnum, 2))

    print("[LOG] count =", count)
    print("[LOG] Done. H5 file has been generated in %s" % KPDROOT)
