import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from keypoint.train_sppe.utils.img import cropBox
from keypoint.train_sppe.torchsample.torchsample.transforms import SpecialCrop, Pad


def draw_heatmap(heatmaps, save_dir):
    """Draw heatmaps of images

    Args
    - heatmap: (torch.Tensor) With size []
    - save_dir: (str)
    """
    heatmaps = heatmaps.cpu().numpy()
    from IPython import embed
    embed()
    for i in range(heatmaps.shape[0]):
        fig, ax = plt.subplots()
        ax.axis('off')
        plt.imshow(heatmaps[i], cmap='jet', interpolation='nearest')
        plt.savefig(os.path.join(save_dir, '%d.png' % i))


def draw_keypoints(img_path, gt_kps, pred_kps, bbox, scores, save_path):
    """Draw keypoints on cropped image

    Args
    - img_path: (str)
    - gt_kps: (np.array) [N x 2]
    - pred_kps: (np.array) [N x 2]
    - bbox: (np.array) [4]
    - scores: (np.array) [N]
    - save_path: (str)
    """
    PAD = 30

    img = Image.open(img_path)
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD
    cropped_img = img.crop((x1, y1, x2, y2))

    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    ax.axis('off')
    plt.imshow(cropped_img)

    max_idx = np.argsort(scores)[::-1]
    good_pred_kps = pred_kps[max_idx[:10]]
    good_gt_kps = gt_kps[max_idx[:10]]

    # Draw good points
    # for i in range(10):
    #     plt.plot((good_pred_kps[i, 0] - x1, good_gt_kps[i, 0] - x1),
    #              (good_pred_kps[i, 1] - y1, good_gt_kps[i, 1] - y1),
    #              c='aqua', linewidth=1, linestyle='--')
    plt.scatter(good_pred_kps[:, 0] - x1,
                good_pred_kps[:, 1] - y1, c='aqua', s=20, marker='x')
    plt.scatter(good_gt_kps[:, 0] - x1,
                good_gt_kps[:, 1] - y1, c='yellow', s=20)

    # Draw bad points
    # bad_pred_kps = pred_kps[max_idx[-10:]]
    # bad_gt_kps = gt_kps[max_idx[-10:]]
    # for i in range(10):
    #     plt.plot((bad_pred_kps[i, 0] - x1, bad_gt_kps[i, 0] - x1),
    #              (bad_pred_kps[i, 1] - y1, bad_gt_kps[i, 1] - y1),
    #              c='aqua', linewidth=1, linestyle='--')
    # plt.scatter(bad_pred_kps[:, 0] - x1, bad_pred_kps[:, 1] - y1, c='yellow', s=3)
    # plt.scatter(bad_gt_kps[:, 0] - x1, bad_gt_kps[:, 1] - y1, c='aqua', s=3)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    img.close()


def draw_6d_pose(img_path, gt_pose, pred_pose, model, cam, save_path):
    """
    Draw a 3d bounding box on image to visualize 6d pose

    Args
    - img_path: (str) path to original image
    - gt_pose: (np.array) [4 x 4] pose matrix
    - pred_pose: (np.array) [4 x 4] pose matrix
    - model: (np.array) [N x 3] model 3d vertices
    - cam: (np.array) [3 x 3] camera matrix
    - save_path: (str) Path to save plots
    """
    img = Image.open(img_path)

    gt_pose = gt_pose[:3]
    pred_pose = pred_pose[:3]

    corners_vertices = get_3D_corners(model)
    gt_corners = project_vertices(corners_vertices, gt_pose, cam)
    pred_corners = project_vertices(corners_vertices, pred_pose, cam)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(np.array(img))
    edges_corners = (
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    )
    ax.scatter(gt_corners[:, 0], gt_corners[:, 1], s=10, c='aqua')
    for edge in edges_corners:
        ax.plot(gt_corners[edge, 0], gt_corners[edge, 1],
                linewidth=1.0, c='aqua')

    ax.scatter(pred_corners[:, 0], pred_corners[:, 1], s=10, c='yellow')
    for edge in edges_corners:
        ax.plot(pred_corners[edge, 0], pred_corners[edge, 1],
                linewidth=1.0, c='yellow')

    # plt.savefig(save_path, bbox_inches='tight', dpi=400)
    # if not os.path.exists(save_path[0:-8]):
    #     os.mkdir(save_path[0:-8])
    #     print(save_path)
    plt.savefig(save_path)

    img.close()
    plt.close()


def get_3D_corners(vertices):
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


def project_vertices(vertices, pose, cam, offset=0):
    """Project 3d vertices to 2d

    Args
    - vertices: (np.array) [N x 3] 3d vertices
    - pose: (np.array) [4 x 4] pose matrix
    - cam: (np.array) [3 x 3] camera matrix

    Returns
    - projected: (np.array) [N x 2] projected 2d points
    """
    vertices = np.concatenate(
        (vertices, np.ones((vertices.shape[0], 1))), axis=1)
    projected = np.matmul(np.matmul(cam, pose), vertices.T)
    projected /= projected[2, :]
    projected = projected[:2, :].T
    return projected


def crop_from_dets(img, box, inputResH, inputResW):
    """
    Crop human from origin image according to Dectecion Results

    Args
    - img: (Tensor) With size [C, W, H]
    - box: (Tensor) With size [4, ]
    """
    _, imght, imgwidth = img.size()#480,640
    tmp_img = img.clone()#origin_img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)

    upLeft = torch.Tensor((float(box[0]), float(box[1])))
    bottomRight = torch.Tensor((float(box[2]), float(box[3])))

    ht = bottomRight[1] - upLeft[1]
    width = bottomRight[0] - upLeft[0]
    if width > 100:
        scaleRate = 0.2
    else:
        scaleRate = 0.3
    #make sure that the coordinate is in area.
    upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
    upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
    bottomRight[0] = max(min(imgwidth - 1, bottomRight[0] +
                             width * scaleRate / 2), upLeft[0] + 5)
    bottomRight[1] = max(min(imght - 1, bottomRight[1] +
                             ht * scaleRate / 2), upLeft[1] + 5)

    inps = cropBox(tmp_img.cpu(), upLeft, bottomRight, inputResH, inputResW)
    pt1 = upLeft
    pt2 = bottomRight

    return inps, pt1, pt2
