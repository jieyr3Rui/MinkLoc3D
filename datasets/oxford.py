# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import os
import pickle
import numpy as np
import math
from scipy.linalg import expm, norm
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import tqdm


class OxfordDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """
    def __init__(self, dataset_path: str, query_filename: str, image_path: str = None,
                 lidar2image_ndx=None, transform=None, set_transform=None, image_transform=None, use_cloud=True):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        self.image_path = image_path
        self.lidar2image_ndx = lidar2image_ndx
        self.image_transform = image_transform
        self.n_points = 4096    # pointclouds in the dataset are downsampled to 4096 points
        self.image_ext = '.png'
        self.use_cloud = use_cloud
        print('{} queries in the dataset'.format(len(self)))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc = self.load_pc(file_pathname)
        # get_item用的是transform
        if self.transform is not None:
            query_pc = self.transform(query_pc)

        return query_pc, ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_path, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        return pc


class TrainingTuple:
    # 一个保存的训练单元
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class ValTransform:
    """
    # 在测试时单个点云进行transform操作
    只作so3的随机旋转
    调用在eval/evaluate.py中的get_latent_vector函数
    """
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            # t = [RandomRotation(), JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
            #      RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]

            t = [
                # so3旋转
                RandomRotation(),
                # z旋转
                # RandomRotation(max_theta2=0, axis=np.array([0, 0, 1])),
            ]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

class TrainTransform:
    """
    # 在训练时对单个点云进行transform操作\n
    调用在datasets/oxford.py的OxfordDataset的__getitem__函数\n
    """
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [
                # z轴旋转
                # RandomRotation(axis=np.array([0,0,1]), max_theta2=0), 
                # 绕z轴旋转
                RandomRotation(axis=np.array([0,0,1]), max_theta=180, max_theta2=0),
                # 绕x轴随机旋转10度以内
                RandomRotation(axis=np.array([1,0,0]), max_theta=10, max_theta2=0), 
                # 绕y轴随机旋转10度以内
                RandomRotation(axis=np.array([0,1,0]), max_theta=10, max_theta2=0), 
                JitterPoints(sigma=0.001, clip=0.002), 
                RemoveRandomPoints(r=(0.0, 0.1)),
                RandomTranslation(max_delta=0.01), 
                RemoveRandomBlock(p=0.4)
            ]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainSetTransform:
    """
    # 对一整个batch的点云作相同的transform
    注意与TrainTransform的区别
    调用在datasets/dataset_utils.py的collate_fn函数
    """
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        # aug_mode是没用上的，代表这个TrainSetTransform一定会应用在整个batch上
        self.aug_mode = aug_mode
        # self.transform = None
        # 原有的方法，适合PointNetVLAD
        # 旋转只在z轴旋转 翻转只在x y轴翻转
        # 强烈说明了MinkLoc3D只适合地地回环，即点云的旋转只绕z轴的情况
        # t = [RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
        #      RandomFlip([0.25, 0.25, 0.])]
        # 模拟地空回环情况
        # 旋转在3个自由度均有发生
        # MinkLoc在这种情况下表现一般
        t = [
            # z轴旋转
            # RandomRotation(axis=np.array([0,0,1]), max_theta2=0), 
            # so3旋转
            # RandomRotation(),
            RandomFlip([0.25, 0.25, 0])
        ]  

        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RandomFlip:
    """
    # 随机翻转点云\b
    p = [p_x, p_y, p_z] 是三个轴翻转的概率\b
    """
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    """
    # 随机旋转点云
    axis=None 指定的旋转轴 默认为None
    max_theta=180 按指定旋转轴的最大旋转角度
    max_theta2=15 按随机旋转轴的最大旋转角度
    """
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        """
        # 计算axis和theta下的旋转矩阵\n
        axis为3维向量，取其归一化值，再乘以旋转角度theta，因此axis / norm(axis) * theta为一个旋转向量。\n
        numpy.cross 返回两个（数组）向量的叉积。\n
        expm 使用Pade近似计算矩阵指数。\n
        """
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            # python的 @ 除了用在装饰器上，还可以用在矩阵操作。
            # 效果大概等同于mul，相当于矩阵乘法。
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    """
    # 随机平移点云
    """
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    """
    # 随机抖动点云
    """
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords


