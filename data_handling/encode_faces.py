import cv2
import glob
from recognition import arcface
from utils import ROOT_DIR
def encode_faces(source_dir=None, image_list=None, encoder = "arcface"):
    """Encodes the images, either by encoding all images loaded from source_dir, or directly from image_list.\n
    Either source_dir or image_list has to be specified!
    Args:
        source_dir (string, optional): source directory, where to load the patches from. Defaults to None.
        image_list (list, optional): image list to directly access images instead of loading patches. Defaults to None.
        
    Returns:
        touple(list, list): (facial images, encodings) with matching indices
    """
    if source_dir is None and image_list is None:
        raise RuntimeError("You need to specify either the image directory or input the list of images for facial encoding.")
    encodings = []
    encoder = arcface.ArcFaceR100(f'{ROOT_DIR}/data/model_files/arcface_r100.pth')
    
    if image_list is None:
        # load images from source_dir
        image_list = []
        for image_path in glob.glob(f'{source_dir}/*.png'):
            patch = cv2.imread(image_path)[:, :, ::-1]
            image_list.append(patch)
    for patch in image_list:
        encoding = encoder.encode(patch)
        encodings.append(encoding)
    return (image_list, encodings)
        