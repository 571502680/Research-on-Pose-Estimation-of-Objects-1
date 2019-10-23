import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from detect.eval.src.config import class_name
from data.linemod.sixd import SixdToolkit


class LinemodDataset(torch.utils.data.dataset.Dataset):
    """LINEMOD dataset"""

    def __init__(self, root, seq, listfile, transform):
        """
        Args
        - root: (str) Path to SIXD dataset test images root
        - seq: (str) LINEMOD sequence number
        - listfile: (str) Listfile
        - transform: (torchvision.transforms)
        """
        self.root = root
        self.transform = transform

        with open(listfile) as f:
            lists = f.readlines()
            lists = [x.strip() for x in lists]

        self.imgs_path = [os.path.join(root, seq, 'rgb', idx + '.png')
                          for idx in lists]

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label_path = img_path.replace(
            'rgb', 'annots/bbox').replace('.png', '.npy')
        img = Image.open(img_path)
        width, height = img.size
        img_tensor = self.transform(img)
        img_label = torch.Tensor(np.load(label_path))
        meta = {
            'path': img_path,
            'width': width,
            'height': height
        }
        return img_tensor, img_label, meta

    def __len__(self):
        return len(self.imgs_path)


class YcbDataset(torch.utils.data.dataset.Dataset):
    """ YCB dataset """

    def __init__(self, seq, root, listfile, transform):
        """
        Args
        - root: (str) Path to YCB dataset test images root
        - listfile: (str) Listfile
        - transform: (torchvision.transforms)
        """
        self.root = root
        self.transform = transform
        self.seq = seq
        self.name = class_name(seq)

        with open(listfile) as f:
            lists = f.readlines()
            lists = [x.strip() for x in lists]

        self.imgs_path = [os.path.join(root, idx + '.png') for idx in lists]

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label_path = img_path.replace('.png', '.txt')

        with open(label_path) as f:
            labels = [x.strip() for x in f.readlines()]

        img = Image.open(img_path)
        width, height = img.size
        img_tensor = self.transform(img)
        meta = {
            'path': img_path,
            'width': width,
            'height': height
        }
        return img_tensor, labels, meta

    def __len__(self):
        return len(self.imgs_path)


def prepare_dataset(name, reso, bs, seq=None, distributed=False):
    """
    Args
    - name: (str) Dataset name
    - reso: (int) Image resolution
    - bs: (int) Batch size
    - seq: (str, optional) Sequence number for linemod
    """
    LINEMOD = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/test'
    YCB = '/home/penggao/data/ycb/'

    train_transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ToTensor()
    ])

    if name == 'linemod-single':
        val_datasets = LinemodDataset(
            root=LINEMOD,
            seq=seq,
            listfile=os.path.join(LINEMOD, seq, 'val.txt'),
            transform=val_transform
        )
    elif name == 'linemod-occ':
        val_datasets = LinemodDataset(
            root=LINEMOD,
            seq='02',
            listfile=os.path.join(LINEMOD, '02', 'val.txt'),
            transform=val_transform
        )
    elif name == 'ycb':
        val_datasets = YcbDataset(
            root=YCB,
            seq=seq,
            listfile=os.path.join(YCB, 'image_sets/test_data_list.txt'),
            transform=val_transform
        )
    else:
        raise NotImplementedError

    if distributed:
        sampler = DistributedSampler(val_datasets)

        val_dataloder = torch.utils.data.DataLoader(
            dataset=val_datasets,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=sampler
        )
    else:
        val_dataloder = torch.utils.data.DataLoader(
            dataset=val_datasets,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    return None, val_dataloder
