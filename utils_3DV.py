# do not move this file --> needed for ROOT Variable
from datetime import datetime
import os
import shutil
# global constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTORS = ["scrfd", "yolo5"]
CLASSIFIERS = ["online", "vgg", "meanshift"]
#usefull helper function
def get_current_datetime_as_str():
    return "{:%Y_%m_%d_%H_%M}".format(datetime.now())

def create_dir(target_dir):
    if target_dir is None:
        target_dir = f"{ROOT_DIR}/data/reconstructions/{get_current_datetime_as_str()}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

# taken from https://stackoverflow.com/questions/19747408/how-get-number-of-subfolders-and-folders-using-python-os-walks
def count_folders(path):
    count1 = 0
    for _, dirs, _ in os.walk(path):
            count1 += len(dirs)
    return count1