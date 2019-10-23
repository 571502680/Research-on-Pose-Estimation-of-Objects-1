import os
import copy
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(description='Keypoints generator')
    # parser.add_argument('--dataset', default='hinterstoisser',
    #                     type=str, help="dataset name")
    parser.add_argument('--sixdroot', default='/home/common/liqi/data/LINEMOD_6D/LM6d_origin/',
                        type=str, help="Path to SIXD root")
    parser.add_argument('--num', default=17,type=int, help="Number of keypoints")
    parser.add_argument('--type', default='cluster',choices=['sift', 'random', 'cluster', 'corner'],
                        type=str, help="Type of keypoints")
    return parser.parse_args()


def get_3d_corners(vertices):
    """Get vertices 3D bounding boxes

    Args
    - vertices: (np.array) [N x 3] 3d vertices

    Returns
    - corners: (np.array) [8 x 2] 2d vertices
    """
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    return corners


if __name__ == '__main__':
    args = parse_args()
    assert args.type is not None, "Please specify type of keypoints"
    assert args.num is not None, "Please specify number of keypoints"
    assert args.type != 'sift', "Please go to ./pcl-sift to generate sift keypoints"

    if args.type == 'corner':
        assert args.num == 9, "Number of \"corner\" keypoints must be 9"

    print("[LOG] Number of keypoints: %d" % args.num)
    print("[LOG] Type of keypoints: %s" % args.type)

    MODEL_ROOT = os.path.join(args.sixdroot,  'models')
    KP_ROOT = os.path.join(args.sixdroot,
                           'kps', str(args.num), args.type)
    if not os.path.exists(KP_ROOT):
        os.makedirs(KP_ROOT)
    else:
        print("[WARNING] Overwrite existing files!")

    tbar = tqdm(os.listdir(MODEL_ROOT))

    for filename in tbar:
        if '.ply' not in filename:  # skip models_info.yml
            continue
        tbar.set_description(filename)

        vertex = np.zeros(args.num,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        model = PlyData.read(os.path.join(MODEL_ROOT, filename))
        xyz = np.stack((np.array(model['vertex']['x']),
                        np.array(model['vertex']['y']),
                        np.array(model['vertex']['z'])), axis=1)

        if args.type == 'random':
            selected_ids = np.random.choice(
                model['vertex'].count, args.num, replace=False)
            for i in range(args.num):
                vertex[i][0] = model['vertex']['x'][selected_ids[i]]
                vertex[i][1] = model['vertex']['y'][selected_ids[i]]
                vertex[i][2] = model['vertex']['z'][selected_ids[i]]
        elif args.type == 'cluster':
            kmeans = KMeans(n_clusters=args.num, max_iter=1000).fit(xyz)
            dist = kmeans.transform(xyz)
            selected_ids = []
            for i in range(args.num):
                di = dist[:, i]
                ind = np.argsort(di)[0]
                vertex[i][0] = model['vertex']['x'][ind]
                vertex[i][1] = model['vertex']['y'][ind]
                vertex[i][2] = model['vertex']['z'][ind]
        elif args.type == 'corner':
            corners = get_3d_corners(xyz)
            center = corners.mean(axis=0).reshape(1,3)
            corners_and_center = np.concatenate((corners, center), axis=0)
            for i in range(args.num):
                vertex[i][0] = corners_and_center[i, 0]
                vertex[i][1] = corners_and_center[i, 1]
                vertex[i][2] = corners_and_center[i, 2]

        data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)

        with open(os.path.join(KP_ROOT, filename), mode='wb') as f:
            data.write(f)
