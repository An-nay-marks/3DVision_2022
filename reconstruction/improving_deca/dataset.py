import os
import torch
import numpy as np
import cv2

from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset
from torch.utils.data import dataset
from utils_3DV import ROOT_DIR


class WeightingPatchesDataset(dataset.Dataset):
    def __init__(self, img_paths, gt_path, preprocessing=None):
        self.img_paths = img_paths
        self.gt_paths = gt_path
        # self.gt = np.loadtxt(gt_path)
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # grab the image path from index
        img_path = self.img_paths[idx]
        # load the image from disk
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255)
        gt = self.gt[idx]
        # check to see if we are applying any transformations
        if self.preprocessing is not None:
            # apply the transformations to both image and its mask
            image = self.preprocessing(x=img)
        # return a tuple of the image and its mask
        return img, gt


class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        folder = f'{ROOT_DIR}/data/NoW_Dataset'
        self.data_path = os.path.join(folder, 'final_release_version', 'imagepathsvalidation.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'final_release_version', 'iphone_pictures')
        self.bbxfolder = os.path.join(folder, 'final_release_version', 'detected_face')

        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        # self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        # self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.crop_size = crop_size
        self.scale = scale

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip())  # + '.jpg'
        bbx_path = os.path.join(self.bbxfolder, self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left'];
        right = bbx_data['right']
        top = bbx_data['top'];
        bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:, :, :3]

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
