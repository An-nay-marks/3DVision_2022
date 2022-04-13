from pipeline import online_pipeline, offline_pipeline
from classify import get_classification_parser, initialize_detector, initialize_classifier
from reconstruction.deca import DECAFaceReconstruction
from utils_3DV import ROOT_DIR, get_default_parser, get_default_objects


def parse_args(online):
    parser = get_classification_parser(online)
    if not online:
        parser.add_argument('-lc', '--load-classified')
    return parser.parse_known_args()[0]


def initialize_deca():
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'
    return DECAFaceReconstruction(deca_file, flame_file, albedo_file)


def run_reconstruction(video_provider, target_dir, online, specific_args):
    online_status = 'online' if online else 'offline'
    print(f'Running {online_status} reconstruction pipeline')

    detector = initialize_detector(specific_args.detector)
    classifier = initialize_classifier(specific_args.classifier)
    deca = initialize_deca()
    pipeline = online_pipeline if online else offline_pipeline
    pipeline.run(video_provider, target_dir, detector, classifier, deca)


def main(default_args):
    run_reconstruction(*get_default_objects(default_args), parse_args(default_args.online))


if __name__ == '__main__':
    main(get_default_parser().parse_known_args()[0])
