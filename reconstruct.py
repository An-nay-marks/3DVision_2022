from classify import get_classification_parser, initialize_detector, initialize_classifier
from pipeline.online_pipeline import OnlinePipeline
from pipeline.offline_pipeline import OfflinePipeline
from reconstruction.deca import DECAFaceReconstruction
from utils_3DV import *


def parse_args(online):
    parser = get_classification_parser(online)
    if not online:
        parser.add_argument('--merge', choices=MERGE_STRATEGIES, default=MERGE_STRATEGIES[0],
                            help=f'Method with which to combine face reconstructions.')
        parser.add_argument('-lc', '--load-classified')
    return parser.parse_known_args()[0]


def initialize_deca(merge_fn):
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'

    if not os.path.exists(albedo_file):
        print('WARNING: Albedo File not found. Reconstruction will be performed without albedo.', file=sys.stderr)
        albedo_file = None

    return DECAFaceReconstruction(deca_file, flame_file, albedo_file, merge_fn)


def run_reconstruction(source, run_name, online, specific_args):
    online_status = 'online' if online else 'offline'
    print(f'Running {online_status} reconstruction pipeline')

    if online or specific_args.load_classified is None:
        classifier = initialize_classifier(specific_args.classifier)
        if online or specific_args.load_raw is None:
            data_src = initialize_video_provider(source)
            detector = initialize_detector(specific_args.detector)
        else:
            data_src = specific_args.load_raw
            detector = None
    else:
        data_src = specific_args.load_classified
        detector = None
        classifier = None

    deca = initialize_deca('single' if online else specific_args.merge)
    pipeline = OnlinePipeline if online else OfflinePipeline
    pipeline = pipeline(data_src, run_name, specific_args.patch_size, detector, classifier, deca)
    pipeline.run()


def main(default_args):
    run_reconstruction(*get_default_objects(default_args), parse_args(default_args.online))


if __name__ == '__main__':
    main(get_default_parser().parse_known_args()[0])
