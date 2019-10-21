import numpy as np

def ADD_err(gt_pose, est_pose, model):
    def transform_points(points_3d, mat):
        rot = np.matmul(mat[:3, :3], points_3d.T)
        return rot.transpose() + mat[:3, 3]
    v_A = transform_points(model, gt_pose)
    v_B = transform_points(model, est_pose)
    v_A = np.array([x for x in v_A])
    v_B = np.array([x for x in v_B])
    return np.mean(np.linalg.norm(v_A - v_B, axis=1))

def ADDS_err(gt_pose, est_pose, model):
    def transform_points(points_3d, mat):
        rot = np.matmul(mat[:3, :3], points_3d.T)
        return rot.transpose() + mat[:3, 3]

    v_A = transform_points(model, gt_pose)
    v_B = transform_points(model, est_pose)
    v_A = np.array([x for x in v_A])
    v_B = np.array([x for x in v_B])

    dist = []
    idxs = np.random.randint(0, v_A.shape[0], 500)
    for idx in idxs:
        va = v_A[idx]
        dist.append(np.linalg.norm(va - v_B, axis=1).min())
    return np.mean(dist)


def rot_error(gt_pose, est_pose):
    def matrix2quaternion(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    gt_quat = Quaternion(matrix2quaternion(gt_pose[:3, :3]))
    est_quat = Quaternion(matrix2quaternion(est_pose[:3, :3]))

    return np.abs((gt_quat * est_quat.inverse).degrees)


def trans_error(gt_pose, est_pose):
    trans_err_norm = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    trans_err_single = np.abs(gt_pose[:3, 3] - est_pose[:3, 3])

    return trans_err_norm, trans_err_single


def IoU(box1, box2):
    """
    Compute IoU between box1 and box2

    Args
    - box: (np.array) bboxes with size [4, ] => [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def projection_error_2d(gt_pose, est_pose, model, cam):
    """
    Compute 2d projection error

    Args
    - gt_pose: (np.array) [4 x 4] pose matrix
    - est_pose: (np.array) [4 x 4] pose matrix
    - model: (np.array) [N x 3] model 3d vertices
    - cam: (np.array) [3 x 3] camera matrix
    """
    gt_pose = gt_pose[:3]
    est_pose = est_pose[:3]
    model = np.concatenate((model, np.ones((model.shape[0], 1))), axis=1)

    gt_2d = np.matmul(np.matmul(cam, gt_pose), model.T)
    est_2d = np.matmul(np.matmul(cam, est_pose), model.T)

    gt_2d /= gt_2d[2, :]
    est_2d /= est_2d[2, :]
    gt_2d = gt_2d[:2, :].T
    est_2d = est_2d[:2, :].T

    return np.mean(np.linalg.norm(gt_2d - est_2d, axis=1))
