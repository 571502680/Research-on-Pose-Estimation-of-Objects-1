import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sixd import SixdToolkit

opj = os.path.join

LINEMOD = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/test'  # LINEMOD data root
DARKNET = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/data/'  # Darknet data root
NAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup',
         'driller', 'duck', 'eggbo', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')


def parse_arg():
    parser = argparse.ArgumentParser(description='Prepare darknet data')
    parser.add_argument('--seq', default='15',type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    print("[LOG] Preparing LINEMOD images for darknet training")
    print("[LOG] Sequence: %s %s" % (args.seq, NAMES[int(args.seq) - 1]))

    # object detection is not related with keypoints number and type
    # so here take 17 and sift as default
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='cluster', is_train=False)

    LINEMOD = opj(LINEMOD, args.seq)
    DARKNET = opj(DARKNET, args.seq)
    SEQNAME = NAMES[int(args.seq) - 1]
    WIDTH = 640
    HEIGHT = 480

    if os.path.exists(DARKNET):
        # print("[WARNING] Overwriting existing images")
        # print("[WARNING] Proceed (y/[n])?", end=' ')
        # choice = input()
        # if choice == 'y':
        os.system('rm -r %s' % DARKNET)
        # else:
        #     print("[LOG] Terminated by user, exit")
        #     sys.exit()

    os.makedirs(DARKNET)
    os.makedirs(opj(DARKNET, 'images'))

    print("[LOG] Preparing data")
    with open(opj(LINEMOD, 'train.txt'), 'r') as f:
        trainlists = f.readlines()
    trainlists = [x.strip() for x in trainlists]
    # with open(opj(LINEMOD, 'test.txt'), 'r') as f:
    #     vallists = f.readlines()
    # vallists = [x.strip() for x in vallists]
    alllists = [x.split('.')[0] for x in os.listdir(opj(LINEMOD, 'rgb'))]
    vallists = list(set(alllists) - set(trainlists))

    tbar = tqdm(alllists)
    for idx in tbar:
        tbar.set_description(idx)
        f = bench.frames[args.seq][int(idx)]

        annot = f['annots'][f['obj_ids'].index(int(args.seq))]
        bbox = annot['bbox']
        xc = ((bbox[2] + bbox[0]) / 2) / WIDTH
        yc = ((bbox[3] + bbox[1]) / 2) / HEIGHT
        w = (bbox[2] - bbox[0]) / WIDTH
        h = (bbox[3] - bbox[1]) / HEIGHT

        with open(opj(DARKNET, 'images/%s.txt' % idx), 'w') as f:
            f.write('0 %f %f %f %f' % (xc, yc, w, h))

        # images
        srcimg = opj(LINEMOD, 'rgb/%s.png' % idx)
        dstimg = opj(DARKNET, 'images/%s.png' % idx)
        os.symlink(srcimg, dstimg)

    with open(opj(DARKNET, 'train.txt'), 'w') as f:
        f.writelines([opj(DARKNET, 'images/%s.png\n' % x) for x in trainlists])

    with open(opj(DARKNET, 'val.txt'), 'w') as f:
        f.writelines([opj(DARKNET, 'images/%s.png\n' % x) for x in vallists])

    with open(opj(DARKNET, '%s.names' % SEQNAME), 'w') as f:
        f.write(SEQNAME)

    DATA = DARKNET.split('train_yolo/')[-1]
    with open(opj(DARKNET, '%s.data') % SEQNAME, 'w') as f:
        f.write('classes=1\n')
        f.write('train=%s/train.txt\n' % DATA)
        f.write('valid=%s/val.txt\n' % DATA)
        f.write('names=%s/%s.names\n' % (DATA, SEQNAME))
        f.write('backup=%s\n' % DATA.replace('data', 'backup'))

    os.makedirs(DATA.replace('data', 'backup'), exist_ok=True)

    print("[LOG] Done")
