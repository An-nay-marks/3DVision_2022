from utils_3DV import *
from pipeline import online_pipeline, offline_pipeline
from classify import get_classification_parser, initialize_detector, initialize_classifier
from reconstruction.deca import DECAFaceReconstruction


def parse_args(online):
    parser = get_classification_parser(online)
    if not online:
        parser.add_argument('-op', '--optimizer', choices=OPTIMIZERS, default=None,
                        help=f'Optimizer to improve the reconstruction, default is None. Can only be called for the offline pipeline')
        parser.add_argument('-lc', '--load-classified')
    return parser.parse_known_args()[0]


def initialize_deca(optimize = None):
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'
    if not os.path.exists(albedo_file):
        print("WARNING: Albedo File not found. Reconstruction will be performed without albedo.")
        albedo_file = None
    return DECAFaceReconstruction(deca_file, flame_file, albedo_file, optimize)


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
    
    if online and specific_args.optimizer is not None:
        print("WARNING: Optimizers can only be run for the offline pipeline. Optimizer will be ignored...")
    deca = initialize_deca(specific_args.optimizer)
    pipeline = online_pipeline if online else offline_pipeline
    pipeline.run(data_src, run_name, specific_args.patch_size, detector, classifier, deca)
    


def main(default_args):
    run_reconstruction(*get_default_objects(default_args), parse_args(default_args.online))


if __name__ == '__main__':
    main(get_default_parser().parse_known_args()[0])
