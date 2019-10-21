import os
import sys
import h5py
import argparse
import numpy as np
from tqdm import tqdm
opj = os.path.join

from sixd import SixdToolkit

YOLOPOSE = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/data_2/LINEMOD'
LINEMOD = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin'
CLASS_NAMES = ('ape', 'benchvise', 'bowl', 'cam', 'can', 'cat', 'cup',
               'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')


def split():
    for seq in os.listdir(opj(LINEMOD, 'test')):
        seq_name = CLASS_NAMES[int(seq) - 1]
        total = os.listdir(opj(LINEMOD, 'test', seq, 'rgb'))
        if os.path.exists(opj(YOLOPOSE, seq_name)):
            yolo_trainfile = opj(YOLOPOSE, seq_name, 'train.txt')
            with open(yolo_trainfile, 'r') as f:
                content = f.readlines()
            content = ['%04d\n' % int(x.split('/')[-1].split('.')[0])
                       for x in content]
            linemod_trainfile = opj(LINEMOD, 'test', seq, 'train.txt')
            with open(linemod_trainfile, 'w') as f:
                f.writelines(content)
        else:
            num = len(total)
            trainlists = np.random.choice(num, int(num/10), replace=False)
            content = ['%04d\n' % x for x in trainlists]
            linemod_trainfile = opj(LINEMOD, 'test', seq, 'train.txt')
            with open(linemod_trainfile, 'w') as f:
                f.writelines(content)


def parse_arg():
    parser = argparse.ArgumentParser(description='Synthetic data generator')
    parser.add_argument('--seq', default='15',type=str)
    parser.add_argument('--kpnum', default=17, choices=[17, 9],
                        type=int, help="Number of keypoints")
    parser.add_argument('--kptype', default='cluster', choices=['sift', 'cluster', 'corner'],
                        type=str, help="Type of keypoints")
    parser.add_argument('--sixdroot', type=str, help="LINEMOD data root directory",
                        default='/home/common/liqi/data/LINEMOD_6D/LM6d_origin/test')
    parser.add_argument('--kpdroot', type=str, help="KPD data root directory",
                        default='/home/common/liqi/data/LINEMOD_6D/LM6d_origin/kps/')
    return parser.parse_args()


LINEMODNAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup',
                'driller', 'duck', 'eggbo', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')


if __name__ == '__main__':
    args = parse_arg()
    print("[LOG] Preparing h5 for KPD training")
    print("[LOG] Number of keypoints: %d" % args.kpnum)
    print("[LOG] Type of keypoints: %s" % args.kptype)
    print("[LOG] Sequence: %s %s" % (args.seq, LINEMODNAMES[int(args.seq)-1]))

    bench = SixdToolkit(dataset='hinterstoisser', kpnum=args.kpnum,
                        kptype=args.kptype, is_train=False)
    LINEMOD = opj(args.sixdroot, args.seq)
    KPDROOT = opj(args.kpdroot, str(args.kpnum), args.kptype, args.seq)

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

    bboxes = {'train': [], 'eval': []}
    kps = {'train': [], 'eval': []}
    imgnames = {'train': [], 'eval': []}
    tbar = tqdm(bench.frames[args.seq])
    for idx, f in enumerate(tbar):
        tbar.set_description('%04d' % idx)
        tag = 'train' if '%04d' % idx in trainlists else 'eval'

        # annotation
        annot = f['annots'][f['obj_ids'].index(int(args.seq))]
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

    print("[LOG] Done. H5 file has been generated in %s" % KPDROOT)
