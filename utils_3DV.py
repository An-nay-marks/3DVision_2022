# do not move this file --> needed for ROOT Variable
import argparse
import cv2
import os
import sys
import shutil
import torch

from datetime import datetime
import time


# global constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = f"{ROOT_DIR}/out"
LOGS_DIR = f"{ROOT_DIR}/logs"

DETECTORS = ['scrfd', 'yolo5']
ONLINE_CLASSIFIERS = ['real-time', 'vgg']
OFFLINE_CLASSIFIERS = ['agglomerative', 'dbscan', 'mean-shift', 'vgg']
OPTIMIZERS = ['mean', 'mean_shape']
SESSION_ID = int(time.time() * 1000)
CHECKPOINTS_DIR = os.path.join(LOGS_DIR, "checkpoints")
CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, str(SESSION_ID))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='Path to input file.')
    parser.add_argument('-r', '--run_name', help='All data outputs of the run will be saved in "out/<run_name>" and all logging to "logs/<run_name>".')
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


# useful helper function
def get_current_datetime_as_str():
    return '{:%Y_%m_%d_%H_%M}'.format(datetime.now())


def init_dir(run_name):
    """creates folders for the data output and logs
    """
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    target_out_path = f"{OUT_DIR}/{run_name}"
    if os.path.exists(target_out_path):
        print(f'Target output directory \"{target_out_path}\" already exists and will be overwritten.')
        print('Do you want to proceed (y/n)?')
        response = sys.stdin.readline().rstrip()
        while response not in ['n', 'y']:
            print('Invalid response. Specify \'y\' (yes) or \'n\' (no).')
            response = sys.stdin.readline().rstrip()
        if response != 'y':
            return False
        shutil.rmtree(target_out_path)
    os.makedirs(target_out_path)
    
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    target_logs_path = f"{LOGS_DIR}/{run_name}"
    if os.path.exists(target_logs_path):
        print(f'Target logging directory \"{target_logs_path}\" already exists and will be overwritten.')
        print('Do you want to proceed (y/n)?')
        response = sys.stdin.readline().rstrip()
        while response not in ['n', 'y']:
            print('Invalid response. Specify \'y\' (yes) or \'n\' (no).')
            response = sys.stdin.readline().rstrip()
        if response != 'y':
            return False
        shutil.rmtree(target_logs_path)
    os.makedirs(target_logs_path)
    return True
