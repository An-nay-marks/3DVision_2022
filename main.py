import utils_3DV

from detect import main as detect
from classify import main as classify
from reconstruct import main as reconstruct


def parse_args():
    parser = utils_3DV.get_default_parser()
    fn_choices = ['detection', 'classification', 'reconstruction']
    parser.add_argument('-f', '--function', choices=fn_choices, default=fn_choices[-1],
                        help=f'Specify the part of the pipeline you would like to run.')
    return parser.parse_known_args()[0]


def main(args):
    if args.function == 'detection':
        detect(args)
    elif args.function == 'classification':
        classify(args)
    elif args.function == 'reconstruction':
        reconstruct(args)


if __name__ == '__main__':
    main(parse_args())
