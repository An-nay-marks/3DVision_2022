import os
import shutil
import itertools
import numpy as np

from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, random_noise
from skimage.transform import estimate_transform, warp, rescale, resize, swirl, PiecewiseAffineTransform
from skimage import exposure
from utils_3DV import ROOT_DIR, check_now_dataset
from tqdm.contrib.concurrent import process_map


def get_augmentations():
    """Downscaling, flipping, transformation, swirling, change in brightness and contrast
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

    def exp_trans(image, g):
        # log (gain) bigger: smaller contrast, gain smaller: bigger contrast: O = gain*log(1 + I)
        return exposure.adjust_gamma(image, g)

    downscales = [0.1, 0.25, 0.05]
    flips = [lambda x:x, lambda x:x[:, ::-1, :]]  # x, -y, color, horizontal flip
    noise = [lambda x:x, lambda x:random_noise(x)]
    swirls = [lambda x:x, lambda x: swirl(x, strength = 2, radius = 100, rotation=0.2)] # not too much of a swirl
    affine_trans = [lambda x:x, lambda x: aff_trans(x)]
    exposures = [lambda x:x, lambda x: exp_trans(x, 0.5),
                 lambda x: exp_trans(x, 0.8), lambda x: exp_trans(x, 1.5)]
    return list(itertools.product(downscales, noise, flips, swirls, affine_trans, exposures))


def get_data_list():
    # copy data list
    head, tail = os.path.split(data.data_path)
    augmented_data_list = os.path.join(head, 'imagepaths_augmented.txt')
    shutil.copy(data.data_path, augmented_data_list)
    return augmented_data_list


def process(idx):
    img_name, img = data[idx]
    for i, (downscale, noise, flip, swir, aff, exp) in enumerate(combined_augmentation):
        downscaled = rescale(img, scale=(downscale, downscale, 1))  # x, y, rgb (don't change rgb)
        noisy = noise(downscaled)
        flipped = flip(noisy)
        swirled = swir(flipped)
        transformed = aff(swirled)
        exposed = exp(transformed)
        resized = resize(exposed, output_shape=img.shape)
        target_img = img_as_ubyte(resized)  # keep uint8 instead of float

        # save new image
        target_name = f"{img_name}_{i:04}"
        target_img_path = os.path.join(data.image_dir, target_name + '.jpg')
        imsave(target_img_path, target_img)

        # copy bbox from original
        bbox_path = os.path.join(data.bbox_dir, img_name + '.npy')
        target_bbox_path = os.path.join(data.bbox_dir, target_name + '.npy')
        shutil.copy(bbox_path, target_bbox_path)

        # save image path to "imagepaths_augmented.txt"
        with open(list_path, 'a') as f:
            f.write(target_name + '\n')


class NoWData:
    def __init__(self):
        self.base_dir = f'{ROOT_DIR}/data/NoW_Dataset/final_release_version'
        self.data_path = os.path.join(self.base_dir, 'imagepathsvalidation.txt')
        self.image_dir = os.path.join(self.base_dir, 'iphone_pictures')
        self.bbox_dir = os.path.join(self.base_dir, 'detected_face')

        with open(self.data_path) as f:
            self.data_lines = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.data_lines[index])  # + '.jpg'
        img = imread(img_path)[:, :, :3]
        img_name = self.data_lines[index].replace('.jpg', '')
        return img_name, img


# multiprocessing needs everything accessible on import -> global namespace
check_now_dataset()
data = NoWData()
combined_augmentation = get_augmentations()
list_path = get_data_list()

if __name__ == "__main__":
    process_map(process, range(len(data)), max_workers=12)
    print("...Done!")
