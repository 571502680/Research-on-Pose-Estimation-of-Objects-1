import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sixd import SixdToolkit

opj = os.path.join


LINEMOD = '/home/penggao/data/sixd/hinterstoisser/test/02'  # LINEMOD data root
DARKNET = '/home/penggao/projects/pose/kp6d/detect/darknet/data/occ'  # Darknet data root
NAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup',
                'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
OCCNAMES = ('ape', 'can', 'cat', 'driller', 'duck',
            'eggbox', 'glue', 'holepuncher')

if __name__ == '__main__':
    print("[LOG] Preparing Occulusion LINEMOD images for darknet training")

    # object detection is not related with keypoints number and type
    # so here take 17 and sift as default
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='sift', is_train=False)

    if os.path.exists(DARKNET):
        print("[WARNING] Overwriting existing images")
        print("[WARNING] Proceed (y/[n])?", end=' ')
        choice = input()
        if choice == 'y':
            os.system('rm -r %s' % DARKNET)
        else:
            print("[LOG] Terminated by user, exit")
            sys.exit()

    WIDTH = 640
    HEIGHT = 480

    os.makedirs(DARKNET)
    os.makedirs(os.path.join(DARKNET, 'images'))

    print("[LOG] Preparing data")
    with open(os.path.join(LINEMOD, 'train.txt'), 'r') as f:
        trainlists = f.readlines()
    trainlists = [x.strip() for x in trainlists]
    alllists = [x.split('.')[0]
                for x in os.listdir(os.path.join(LINEMOD, 'rgb'))]
    vallists = list(set(alllists) - set(trainlists))

    tbar = tqdm(alllists)
    for idx in tbar:
        tbar.set_description(idx)
        f = bench.frames['02'][int(idx)]
        annots = []

        for annot in f['annots']:
            obj_id = annot['obj_id']
            if obj_id == 2:
                continue
            cls_name = NAMES[int(obj_id) - 1]
            occ_id = OCCNAMES.index(cls_name)

            bbox = annot['bbox']
            xc = ((bbox[2] + bbox[0]) / 2) / WIDTH
            yc = ((bbox[3] + bbox[1]) / 2) / HEIGHT
            w = (bbox[2] - bbox[0]) / WIDTH
            h = (bbox[3] - bbox[1]) / HEIGHT

            annots.append('%d %f %f %f %f\n' % (occ_id, xc, yc, w, h))

        with open(os.path.join(DARKNET, 'images/%s.txt' % idx), 'w') as f:
            f.writelines(annots)

        # images
        srcimg = os.path.join(LINEMOD, 'rgb/%s.png' % idx)
        dstimg = os.path.join(DARKNET, 'images/%s.png' % idx)
        os.symlink(srcimg, dstimg)

    with open(opj(DARKNET, 'train.txt'), 'w') as f:
        f.writelines([opj(DARKNET, 'images/%s.png\n' % x) for x in trainlists])

    with open(opj(DARKNET, 'val.txt'), 'w') as f:
        f.writelines([opj(DARKNET, 'images/%s.png\n' % x) for x in vallists])

    with open(opj(DARKNET, 'occ.names'), 'w') as f:
        f.writelines([name + '\n' for name in OCCNAMES])

    DATA = DARKNET.split('train_yolo/')[-1]
    with open(opj(DARKNET, 'occ.data'), 'w') as f:
        f.write('classes=8\n')
        f.write('train=%s/train.txt\n' % DATA)
        f.write('valid=%s/val.txt\n' % DATA)
        f.write('names=%s/occ.names\n' % (DATA))
        f.write('backup=%s\n' % DATA.replace('data', 'backup'))

    os.makedirs(DATA.replace('data', 'backup'), exist_ok=True)

    print("[LOG] Done")
