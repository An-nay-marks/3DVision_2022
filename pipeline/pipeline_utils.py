import os


def pad_face(img, left, top, right, bottom):
    factor = 0.35
    pad_x = round((right - left) * factor / 2)
    pad_y = round((bottom - top) * factor / 2)

    left = max(0, left - pad_x)
    right = min(img.shape[1], right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(img.shape[0], bottom + pad_y)
    return left, top, right, bottom


def create_anonymous_export_dir(target_dir, frame_idx):
    sample_dir = os.path.join(target_dir, f'frame_{frame_idx + 1}')
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir


def create_id_export_dir(target_dir, name):
    sample_dir = os.path.join(target_dir, name)
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir
