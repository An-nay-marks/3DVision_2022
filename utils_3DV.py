# do not move this file --> needed for ROOT Variable
import argparse
import cv2
import os
import sys
import shutil
import torch
import time

from datetime import datetime


# global constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, 'out')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')

DETECTORS = ['scrfd', 'yolo5']
ONLINE_CLASSIFIERS = ['real-time', 'vgg']
OFFLINE_CLASSIFIERS = ['agglomerative', 'dbscan', 'mean-shift', 'vgg']
MERGE_STRATEGIES = ['single', 'mean', 'mean_shape', 'cnn']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='Path to input file.')
    parser.add_argument('-r', '--run-name',
                        help='All data outputs of the run will be saved in "out/<run_name>" '
                             'and all logging to "logs/<run_name>".')
    parser.add_argument('--online', action='store_true')
    return parser


def initialize_video_provider(source):
    if source is None:
        raise ValueError('No source provided!')

    # expand this later to support other sources
    return cv2.VideoCapture(source)


def get_default_objects(args):
    run_name = args.run_name
    if run_name is None:
        run_name = f'{get_current_datetime_as_str()}'

    return args.source, run_name, args.online


def get_current_datetime_as_str():
    return '{:%Y_%m_%d_%H_%M}'.format(datetime.now())


def init_dir(dir_path):
    """
    Initializes empty directory at specified path and ask for confirmation before overwriting existing data.
    """
    if os.path.exists(dir_path):
        print(f'Directory \"{dir_path}\" already exists and will be overwritten.')
        print('Do you want to proceed (y/n)?')
        response = sys.stdin.readline().rstrip()
        while response not in ['n', 'y']:
            print('Invalid response. Specify \'y\' (yes) or \'n\' (no).')
            response = sys.stdin.readline().rstrip()
        if response != 'y':
            return False
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return True


def read_lines_as_list(textfile):
    with open(textfile) as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def check_now_dataset():
    base_dir = os.path.join(ROOT_DIR, 'data', 'NoW_Dataset', 'final_release_version')
    for path in ['scans', 'iphone_pictures', 'detected_face', 'imagepathsvalidation.txt']:
        if not os.path.exists(os.path.join(base_dir, path)):
            raise FileNotFoundError("Please download the NoW evaluation scans and corresponding image, "
                                    "detected face data as well as imagepathsvalidation and put it into "
                                    "data/NoW_Dataset/final_release_version")


def pad_with_zeros(x, string_length):
    str_x = str(x)
    if len(str_x) >= string_length:
        return str_x
    zeros = "".join(["0"*(string_length-len(str_x))])
    return zeros + str_x