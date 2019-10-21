# ------------------------------------------------------------------------------
# Copyright (c) 2018 Gao Peng
# Licensed under the MIT License.
# Written by: Gao PENG (ecr23pg@gmail.com)
# ------------------------------------------------------------------------------

import os
import cv2
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from plyfile import PlyData

opj = os.path.join


class SixdToolkit:
    """Sixd toolkit, load datasets and some normal operations

    Attrs
    - root: (str) Path to root. e.g. '/home/penggao/data/sixd/hinterstoisser'
    - kpnum: (int) Number of keypoints. e.g. 17
    - kptype: (str) Type of keypoints. e.g. 'cluster'
    - pklpath: (str) Path to .pkl file
    - cam: (np.array) [3 x 3] camera matrix
    - models: (dict) Named with sequence number (e.g. '01')
        Each item is a [N x 3] np.array for corresponding model vertices
    - models_info: (dict) Named with sequence number (e.g. '01')
    - kps: (dict) Same format as 'models', represents corresponding keypoints
    - frames: (dict) Named with sequence number (e.g. '01')
          Each item is a list of image frames, with file paths and annotations
    """

    def __init__(self, dataset, kpnum, kptype, is_train, resume=True):
        # Prepare
        self.root = opj('/home/penggao/data/sixd', dataset)
        self.kpnum = kpnum
        self.kptype = kptype
        self.is_train = is_train

        self.pklpath = opj(self.root, 'libs/benchmark.%s-%d-%s.pkl' %
            ('train' if is_train else 'test', self.kpnum, self.kptype))
        self.seq_num = 15

        self.cam = np.zeros((3, 3))
        self.models = dict()
        self.models_info = dict()
        self.kps = dict()
        self.frames = dict()

        # Try to load from disk
        if resume == True:
            try:
                self._load_from_disk()
                print("[LOG] Load SIXD from pkl file success")
                return
            except Exception as e:
                print("[ERROR]", str(e))
                print("[ERROR] Load from pkl file failed. Load all anew")
        else:
            print("[LOG] Load SXID all anew")

        # Load camera matrix
        print("[LOG] Load camera matrix")
        with open(os.path.join(self.root, 'camera.yml')) as f:
            content = yaml.load(f)
            self.cam = np.array([[content['fx'], 0, content['cx']],
                                 [0, content['fy'], content['cy']],
                                 [0, 0, 1]])

        # Load models and keypoints
        print("[LOG] Load models and keypoints")
        MODEL_ROOT = os.path.join(self.root, 'models')
        KP_ROOT = os.path.join(
            self.root, 'kps', str(self.kpnum), self.kptype)
        with open(os.path.join(MODEL_ROOT, 'models_info.yml')) as f:
            content = yaml.load(f)
            for key, val in tqdm(content.items()):
                name = '%02d' % int(key)  # e.g. '01'
                self.models_info[name] = val

                ply_path = os.path.join(MODEL_ROOT, 'obj_%s.ply' % name)
                data = PlyData.read(ply_path)
                self.models[name] = np.stack((np.array(data['vertex']['x']),
                                              np.array(data['vertex']['y']),
                                              np.array(data['vertex']['z'])), axis=1)

                kp_path = os.path.join(KP_ROOT, 'obj_%s.ply' % name)
                data = PlyData.read(kp_path)
                self.kps[name] = np.stack((np.array(data['vertex']['x']),
                                           np.array(data['vertex']['y']),
                                           np.array(data['vertex']['z'])), axis=1)

        # Load annotations
        print("[LOG] Load annotations")
        for seq in tqdm(['%02d' % i for i in range(1, self.seq_num+1)]):
            frames = list()
            seq_root = opj(
                self.root, 'train' if self.is_train else 'test', str(seq))
            imgdir = opj(seq_root, 'rgb')
            with open(opj(seq_root, 'gt.yml')) as f:
                content = yaml.load(f)
                for key, val in content.items():
                    frame = dict()
                    frame['path'] = opj(imgdir, '%04d.png' % int(key))
                    frame['annots'] = list()
                    obj_ids = []
                    for v in val:
                        annot = dict()
                        rot = np.array(v['cam_R_m2c']).reshape(3, 3)
                        tran = np.array(v['cam_t_m2c']).reshape(3, 1)
                        annot['pose'] = np.concatenate((rot, tran), axis=1)
                        # x1 y1 w h => x1 y1 x2 y2
                        bbox = np.array(v['obj_bb'])
                        bbox[2] += bbox[0]
                        bbox[3] += bbox[1]
                        annot['bbox'] = bbox
                        annot['obj_id'] = v['obj_id']
                        annot['kps'] = self.project_vertices(
                            self.kps['%02d' % v['obj_id']], annot['pose'])
                        frame['annots'].append(annot)
                        obj_ids.append(v['obj_id'])
                    frame['obj_ids'] = obj_ids
                    frames.append(frame)
            self.frames[seq] = frames

        # Save to pickle path
        try:
            self._save_to_disk()
            print("[LOG] Save benchmark to disk")
        except Exception as e:
            print("[ERROR]", str(e))
            print("[ERROR] Save to disk failed")

    def project_vertices(self, vertices, pose):
        """Project 3d vertices to 2d

        Args
        - vertices: (np.array) [N x 3] 3d keypoints vertices.
        - pose: (np.array) [3 x 4] pose matrix

        Returns
        - projected: (np.array) [N x 2] projected 2d points
        """
        vertices = np.concatenate(
            (vertices, np.ones((vertices.shape[0], 1))), axis=1)
        projected = np.matmul(np.matmul(self.cam, pose), vertices.T)
        projected /= projected[2, :]
        projected = projected[:2, :].T
        return projected

    def get_3d_corners(self, vertices):
        """Get vertices 3d bounding boxes

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

    def solve_pnp(self, model_points, image_points):
        """
        Wrapper for OpenCV solvePnP

        Args
        - model_points: (np.array) [N x 3] 3d points on model
        - image_points: (np.array) [N x 2] Corresponding 2d points

        Returns
        - pose: (np.array) [3 x 4] Pose matrix

        Docs
        - cv2.solvePnP: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp
        - cv2.Rodrigues: http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20Rodrigues(InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian)
        """
        _, rvec, T, _ = cv2.solvePnPRansac(
            model_points,
            image_points,
            self.cam,
            distCoeffs=np.zeros((4,1))
        )
        R = np.eye(3)
        cv2.Rodrigues(rvec, R)
        pose = np.concatenate((R, T), axis=1)
        return pose

    def _load_from_disk(self):
        assert os.path.exists(self.pklpath) == True, ".pkl file doesn't exist"
        assert os.path.getsize(self.pklpath) > 0, ".pkl file corrupted"
        with open(self.pklpath, 'rb') as handle:
            benchmark = pickle.load(handle)
        assert benchmark['root'] == self.root, "Wrong dataset root, %s v.s. %s" % (
            benchmark['root'], self.root)
        assert benchmark['kpnum'] == self.kpnum, "Wrong number of keypoints, %d v.s. %d" % (
            benchmark['kpnum'], self.kpnum)
        assert benchmark['kptype'] == self.kptype, "Wrong type of keypoints, %s v.s. %s" % (
            benchmark['kptype'], self.kptype)
        assert benchmark['pklpath'] == self.pklpath, "Wrong .pkl path, %s v.s. %s" % (
            benchmark['pklpath'], self.pklpath)
        self.__dict__ = benchmark

    def _save_to_disk(self):
        if os.path.exists(self.pklpath):
            print("[WARNING] Overwrite benchmark")
            os.remove(self.pklpath)
        with open(self.pklpath, 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
