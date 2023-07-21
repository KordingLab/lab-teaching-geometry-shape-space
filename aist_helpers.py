import numpy as np
import torch
import matplotlib.pyplot as plt

re_order_joints = [
    0,
    5, 7, 9,
    6, 8, 10,
    11, 13, 15,
    12, 14, 16,
    1, 2,
    3, 4,
]


def load_poses(file) -> torch.Tensor:
    """Return frames x keypoints x xyz coordinates tensor
    """
    data = np.load(file, allow_pickle=True)
    return torch.tensor(data['keypoints3d'][:, re_order_joints, :])


keypoint_names = [
    'nose',
    'rshoulder', 'relbow', 'rwrist',
    'lshoulder', 'lelbow', 'lwrist',
    'rhip', 'rknee', 'rankle',
    'lhip', 'lknee', 'lankle',
    'reye', 'leye',
    'rear', 'lear'
]

bones = [
    # head
    ('reye', 'leye'),
    ('nose', 'reye'),
    ('nose', 'leye'),
    ('reye', 'rear'),
    ('leye', 'lear'),
    # torso trapezoid
    ('rshoulder', 'lshoulder'),
    ('rshoulder', 'rhip'),
    ('lshoulder', 'lhip'),
    ('rhip', 'lhip'),
    # arms
    ('rshoulder', 'relbow'),
    ('relbow', 'rwrist'),
    ('lshoulder', 'lelbow'),
    ('lelbow', 'lwrist'),
    # legs
    ('rhip', 'rknee'),
    ('rknee', 'rankle'),
    ('lhip', 'lknee'),
    ('lknee', 'lankle'),
]


def plot_bones(pose, ax=None, **kwargs):
    ax = ax if ax is not None else plt.gca()
    for bone in bones:
        start = pose[keypoint_names.index(bone[0])]
        end = pose[keypoint_names.index(bone[1])]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **kwargs)
