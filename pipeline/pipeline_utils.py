import os
import cv2

from tqdm import tqdm


def pad_face(img, left, top, right, bottom):
    factor = 0.6
    pad_x = round((right - left) * factor / 2)
    pad_y = round((bottom - top) * factor / 2)

    left = max(0, left - pad_x)
    right = min(img.shape[1], right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(img.shape[0], bottom + pad_y)
    return left, top, right, bottom


def resize_face(img, export_size):
    if export_size is None:
        return img

    return cv2.resize(img, export_size)


def create_anonymous_export_dir(target_dir, frame_idx):
    sample_dir = os.path.join(target_dir, f'frame_{frame_idx + 1}')
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir


def create_id_export_dir(target_dir, name):
    sample_dir = os.path.join(target_dir, name)
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir


def load_raw_patches(path):
    patches = []
    for directory in tqdm(os.listdir(path)):
        dir_path = os.path.join(path, directory)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            patches.append(cv2.imread(file_path))
    return patches


def load_classified_patches(path):
    patches = []
    identities = []
    for identity in tqdm(os.listdir(path)):
        dir_path = os.path.join(path, identity)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            patches.append(cv2.imread(file_path))
            identities.append(identity)
    return patches, identities


def find_matches(bboxes, bboxes_flipped, matches):
    total_flipped = 0

    #loop over all faces detected in the current frame
    for i in range(0,bboxes.shape[0]):
            #keep track of the minimum difference seen so far between the 'left' 
            #and 'right' values of the bounding box in the original vs. flipped version
            min_left_dif = 10000.0
            min_right_dif = 10000.0

            #for each face detected in the current frame go through all faces detected in the flipped version
            for j in range(0, bboxes_flipped.shape[0]):
                
                #we go through all faces in the flipped version here.
                if i == 0:
                    total_flipped += 1

                left_dif =  abs(bboxes[i][1] - bboxes_flipped[j][1])
                right_dif = abs(bboxes[i][3] - bboxes_flipped[j][3])

                #most likely the same face if the difference is lower than a small threshold
                #also make sure it's the smallest difference seen so far

                if left_dif < min(10.0, min_left_dif) and right_dif < min(20.0, min_right_dif):
                    matches[i] = j
                    min_left_dif = left_dif
                    min_right_dif = right_dif
            
            #print('face ', i, ' in the original frame matched with face ', matches[i], ' in the flipped frame')
    
    #print(matches)
    return matches, total_flipped
