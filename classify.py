from utils_3DV import *
from pipeline.online_pipeline import OnlinePipeline
from pipeline.offline_pipeline import OfflinePipeline
from detect import get_detection_parser, initialize_detector
from recognition import arcface, real_time, vgg, clustering


def get_classification_parser(online):
    parser = get_detection_parser()
    classifiers = ONLINE_CLASSIFIERS if online else OFFLINE_CLASSIFIERS
    parser.add_argument('-c', '--classifier', type=str, choices=classifiers, default=classifiers[0],
                        help=f'Classification model to use, default is {classifiers[0]}.')
    if not online:
        parser.add_argument('-lr', '--load-raw')
    return parser


def parse_args(online):
    parser = get_classification_parser(online)
    return parser.parse_known_args()[0]


def initialize_classifier(model):
    if model == 'vgg':
        return vgg.VGGFaceClassifier(threshold=0.3)

    encoder = arcface.ArcFaceR100(f'{ROOT_DIR}/data/model_files/arcface_r100.pth')

    if model == 'real-time':
        return real_time.RealTimeFaceIdentifier(encoder, threshold=0.3)
    elif model == 'agglomerative':
        return clustering.AgglomerativeFaceClassifier(encoder, threshold=0.3)
    elif model == 'dbscan':
        return clustering.DBSCANFaceClassifier(encoder, threshold=0.5)
    elif model == 'mean-shift':
        return clustering.MeanShiftFaceClassifier(encoder)
    else:
        return None


def run_classification(source, run_name, online, specific_args):
    online_status = 'online' if online else 'offline'
    print(f'Running {online_status} classification pipeline')

    if online or specific_args.load_raw is None:
        data_src = initialize_video_provider(source)
        detector = initialize_detector(specific_args.detector)
    else:
        data_src = specific_args.load_raw
        detector = None

    classifier = initialize_classifier(specific_args.classifier)
    pipeline =  OnlinePipeline if online else OfflinePipeline
    pipeline = pipeline(data_src, run_name, specific_args.patch_size, detector, classifier)
    pipeline.run()


def main(default_args):
    run_classification(*get_default_objects(default_args), parse_args(default_args.online))


if __name__ == '__main__':
    main(get_default_parser().parse_known_args()[0])
