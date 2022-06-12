import os
from glob import glob

import torch
import numpy as np

from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset
from utils_3DV import ROOT_DIR, DEVICE


class NoWDataset(Dataset):
    def __init__(self, split='all', train_ratio=0.8, crop_size=224, scale=1.6, dist_path=None):
        self.scale = scale
        self.crop_size = crop_size

        self.base_dir = f'{ROOT_DIR}/data/NoW_Dataset/final_release_version'
        self.image_dir = os.path.join(self.base_dir, 'iphone_pictures')
        self.bbox_dir = os.path.join(self.base_dir, 'detected_face')
        self.scan_dir = os.path.join(self.base_dir, 'scans')
        self.lmk_dir = os.path.join(self.base_dir, 'scans_lmks_onlypp')

        with open(os.path.join(self.base_dir, 'imagepathsvalidation.txt')) as f:
            self.data_lines = [line.strip().replace('.jpg', '') for line in f.readlines()]

        assert 0 < train_ratio < 1
        train_size = int(len(self.data_lines) * train_ratio)
        if split == 'train':
            self.data_lines = self.data_lines[:train_size]
            self.distances = np.load(dist_path)[:train_size] if dist_path else None
        elif split == 'test':
            self.data_lines = self.data_lines[train_size:]
            self.distances = np.load(dist_path)[train_size:] if dist_path else None
        else:
            self.distances = np.load(dist_path) if dist_path else None

        if self.distances is not None:
            assert len(self.data_lines) == len(self.distances)

    def __len__(self):
        return len(self.data_lines)

    def _load_image(self, img_path, bbox_path):
        bbox_data = np.load(bbox_path, allow_pickle=True, encoding='latin1').item()

        left = bbox_data['left']
        right = bbox_data['right']
        top = bbox_data['top']
        bottom = bbox_data['bottom']

        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = imread(img_path)[:, :, :3] / 255.
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        return torch.tensor(dst_image).float().to(DEVICE)

    def __getitem__(self, index):
        subject = self.data_lines[index].split('/')[0]
        gt_mesh_path = glob(os.path.join(self.scan_dir, subject, '*.obj'))[0]
        gt_lmk_path = glob(os.path.join(self.lmk_dir, subject, '*.pp'))[0]

        img_path = os.path.join(self.image_dir, self.data_lines[index] + 'jpg')
        bbox_path = os.path.join(self.bbox_dir, self.data_lines[index] + 'npy')
        img = self._load_image(img_path, bbox_path)

        if self.distances is None:
            return img, gt_mesh_path, gt_lmk_path,
        else:
            return img, self.distances[index]


class SubjectBasedNoWDataSet(NoWDataset):
    def __init__(self):
        super().__init__()
        self.subjects = os.listdir(self.scan_dir)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        subject = self.subjects[index]
        relevant_lines = [l for l in self.data_lines if subject in l]

        images = []
        gt_mesh_path = glob(os.path.join(self.scan_dir, subject, '*.obj'))[0]
        gt_lmk_path = glob(os.path.join(self.lmk_dir, subject, '*.pp'))[0]

        for line in relevant_lines:
            img_path = os.path.join(self.image_dir, line + 'jpg')
            bbox_path = os.path.join(self.bbox_dir, line + 'npy')
            images.append(self._load_image(img_path, bbox_path))

        return images, gt_mesh_path, gt_lmk_path
