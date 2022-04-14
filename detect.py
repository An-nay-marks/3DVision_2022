from utils_3DV import *
from detection import scrfd, yolo5
from pipeline import online_pipeline, offline_pipeline


def get_detection_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector', choices=DETECTORS, default=DETECTORS[0],
                        help=f'Detector Model, default is {DETECTORS[0]}.')
    return parser


def parse_args():
    parser = get_detection_parser()
    parser.add_argument('--patch-size', type=int, nargs=2,
                        help='Patch size output (width, height), if specific size is required.')
    return parser.parse_known_args()[0]


def initialize_detector(model):
    if model == 'scrfd':
        return scrfd.SCRFaceDetector(f'{ROOT_DIR}/data/model_files/scrfd_34g.onnx')
    elif model == 'yolo5':
        return yolo5.YOLOv5FaceDetector(f'{ROOT_DIR}/data/model_files/yolov5l.pt')
    else:
        return None


def run_detection(source, target_dir, online, specific_args):
    online_status = 'online' if online else 'offline'
    print(f'Running {online_status} detection pipeline')

    provider = initialize_video_provider(source)
    detector = initialize_detector(specific_args.detector)
    pipeline = online_pipeline if online else offline_pipeline
    pipeline.run(provider, target_dir, specific_args.patch_size, detector)


def main(default_args):
    run_detection(*get_default_objects(default_args), parse_args())


if __name__ == '__main__':
    main(get_default_parser().parse_known_args()[0])
