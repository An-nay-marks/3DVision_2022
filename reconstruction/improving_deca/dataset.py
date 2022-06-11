import itertools
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import shutil
from pathlib import Path

from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.transform import estimate_transform, warp, rescale, resize, swirl, PiecewiseAffineTransform
from skimage import exposure
from torch.utils.data import Dataset
from torch.utils.data import dataset
from utils_3DV import ROOT_DIR, check_dataset, pad_with_zeros


def augment_patches(start_counter_offset):
    """Downscaling, flipping, transformation, swirling, change in brightness and contrast

    Args:
        start_counter_offset (int) offset for the FaMoS Number in order to avoid collision with existing data, increment with each call of this function by 1
    """
    def aff_trans(image):
        # adapted from scikit docs
        rows, cols = image.shape[0], image.shape[1]
        src_cols = np.linspace(0, cols, 20)
        src_rows = np.linspace(0, rows, 10)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        # add sinusoidal oscillation to row coordinates
        dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 7
        dst_cols = src[:, 0]
        dst_rows *= 1.5
        dst_rows -= 1.5 * 7
        dst = np.vstack([dst_cols, dst_rows]).T
        aff = PiecewiseAffineTransform()
        aff.estimate(src, dst)
        return warp(image, aff)
    def exp_trans(image, g, l):
        # gamma bigger 1: darker, smaller 1: lighter
        # log (gain) bigger: smaller contrast, gain smaller: bigger contrast: O = gain*log(1 + I) 
        gamma_corrected = exposure.adjust_gamma(image, g)
        log_corrected = exposure.adjust_gamma(gamma_corrected, l)
        return log_corrected
    check_dataset()
    data_set = NoWDataset()
    iphone_pictures_folder = data_set.imagefolder
    detected_face_folder = data_set.bbxfolder
    scans_path = os.path.join(data_set.folder, "scans")
    data_path_txt = data_set.data_path
    base_number = str(200000 + start_counter_offset)
    counter = 1
    counter_len = 5
    img_counter = 1303 + (1303 * 200 * (start_counter_offset)) # there will be approx 1302 * 200 images created per run
    img_counter_len = 6
    downscales = [0.01, 0.05, 0.1, 0.2, 0.5]
    flips = [lambda x:x, lambda x:x[:, ::-1, :]] # x, -y, color, horizontal flip
    swirls = [lambda x:x, lambda x: swirl(x, strength = 2, radius = 100, rotation=0.2)] # not too much of a swirl
    affine_trans = [lambda x:x, lambda x: aff_trans(x)]
    exposures = [lambda x:x, lambda x: exp_trans(x, 0.8, 0.8), lambda x: exp_trans(x, 0.8, 1.5), lambda x: exp_trans(x, 1.5, 0.8), lambda x: exp_trans(x, 2, 1.5), lambda x: exp_trans(x, 1.3, 1.5)]
    
    combined_augmentation = list(itertools.product(downscales, flips, swirls, affine_trans, exposures))
    
    previous_FaMoS = ""
    for idx in tqdm(range(len(data_set))):
        for downscale, flip, swir, aff, exp in combined_augmentation:
            dict = data_set[idx]
            # augment iphone_pictures
            img = dict['original_image']
            img_downscaled = rescale(img, scale = (downscale, downscale, 1)) # x, y, rgb (don't change rgb)
            flipped = flip(img_downscaled)
            swirled = swir(flipped)
            transformed = aff(swirled)
            corrected = exp(transformed)
            target_image = img_as_ubyte(resize(corrected, output_shape = img.shape)) # keep uint8 instead of float
            
            # extract path names and target FaMoS nunmber
            FaMoS_number, expression_type, image_name = dict['imagename'].split('/') # e.g. FaMoS_180618_03331_TA/multiview_neutral/IMG_1216
            target_FaMoS_number = f"FaMoS_{base_number}_{pad_with_zeros(counter, counter_len)}_TA/"
            
            # copy scans
            scan_folder = os.path.join(scans_path, FaMoS_number)
            scan_name = os.listdir(scan_folder)[0] # with file extension already
            scan_path = os.path.join(scan_folder, scan_name)
            target_scan_folder = os.path.join(scans_path, target_FaMoS_number)
            target_scan_path = os.path.join(target_scan_folder, scan_name) # use same scan file name, different folder
            if not os.path.exists(target_scan_folder): # one scan for multiple images
                Path(target_scan_folder).mkdir(parents = True)
                shutil.copyfile(src = scan_path, dst = target_scan_path)
            
            # copy detected face
            target_image_name = f"IMG_{pad_with_zeros(img_counter, img_counter_len)}"
            detected_face_path = os.path.join(detected_face_folder, FaMoS_number, expression_type, f"{image_name}.npy")
            target_detected_face_folder = os.path.join(detected_face_folder, target_FaMoS_number, expression_type)
            target_detected_face_path = os.path.join(target_detected_face_folder, f"{target_image_name}.npy")
            Path(target_detected_face_folder).mkdir(parents = True, exist_ok = True)
            shutil.copyfile(src = detected_face_path, dst = target_detected_face_path)
            
            # save blurred images
            target_image_folder = os.path.join(iphone_pictures_folder, target_FaMoS_number, expression_type)
            target_image_path = os.path.join(target_image_folder, f"{target_image_name}.jpg")
            Path(target_image_folder).mkdir(parents = True, exist_ok = True)
            imsave(target_image_path, target_image)
            
            # increase counters
            if not previous_FaMoS == FaMoS_number: # multiple images might have the same FaMoS number
                if not previous_FaMoS == "": # don't increase in first loop iteration
                    counter += 1
                previous_FaMoS = FaMoS_number
            img_counter += 1
            
            # save data path to "imagespathsvalidation.txt"
            target_line = os.path.join(target_FaMoS_number, f"{expression_type}/{target_image_name}.jpg")
            with open(data_path_txt, 'a') as f:
                f.write(f"{target_line}\n")
    print("...Done!")

# class WeightingPatchesDataset(dataset.Dataset):
#     def __init__(self, img_paths, gt_path, preprocessing=None):
#         self.img_paths = img_paths
#         self.gt_paths = gt_path
#         # self.gt = np.loadtxt(gt_path)
#         self.preprocessing = preprocessing

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         # grab the image path from index
#         img_path = self.img_paths[idx]
#         # load the image from disk
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (224, 224))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.transpose(img, (2, 0, 1))
#         img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
#         img.div_(255)
#         gt = self.gt[idx]
#         # check to see if we are applying any transformations
#         if self.preprocessing is not None:
#             # apply the transformations to both image and its mask
#             image = self.preprocessing(x=img)
#         # return a tuple of the image and its mask
#         return img, gt


class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        self.folder = f'{ROOT_DIR}/data/NoW_Dataset/final_release_version'
        self.data_path = os.path.join(self.folder, 'imagepathsvalidation.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(self.folder, 'iphone_pictures')
        self.bbxfolder = os.path.join(self.folder, 'detected_face')
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

        image_transformed = image / 255.
        dst_image = warp(image_transformed, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                'original_image': image
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
