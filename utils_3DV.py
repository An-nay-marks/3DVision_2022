# do not move this file --> needed for ROOT Variable
import argparse
import cv2
import os
import shutil

from datetime import datetime

# global constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTORS = ['scrfd', 'yolo5']
CLASSIFIERS = ['real-time', 'vgg', 'mean-shift']


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='Path to input file.')
    parser.add_argument('-t', '--target', help='Path to export directory.')
    parser.add_argument('--online', action='store_true')
    return parser


def initialize_video_provider(source):
    if source is None:
        raise ValueError('No source provided!')

    # expand this later to support other sources
    return cv2.VideoCapture(source)


def get_default_objects(args):
    target_dir = args.target
    if target_dir is None:
        target_dir = f'{ROOT_DIR}/out/{get_current_datetime_as_str()}'

    return args.source, target_dir, args.online


# useful helper function
def get_current_datetime_as_str():
    return '{:%Y_%m_%d_%H_%M}'.format(datetime.now())


def init_dir(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
