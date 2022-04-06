import argparse

from data_handling.detect_faces import face_detection
from data_handling.offline_pipeline import run_offline_pipeline
from data_handling.online_pipeline import run_online_pipeline
from utils_3DV import DETECTORS, CLASSIFIERS


def parse_args():
    functions_dic = {'detect_faces': face_detection,
                     'online_pipeline': run_online_pipeline,
                     'offline_pipeline': run_offline_pipeline}

    # parent parser
    parser = argparse.ArgumentParser(description="Implementation of the 3D Face Reconstruction Pipeline using Videos",
                                     add_help=False)
    default_function = list(functions_dic.keys())[0]
    parser.add_argument('-f', '--function', default=default_function, type=str,
                        help=f"Give the function you would like to run. Default = \"{default_function}\".\n"
                             "Currently available:\n".join(functions_dic.keys()))
    function_name = parser.parse_known_args()[0].function.lower()

    # specific parser to each function
    specific_parser = argparse.ArgumentParser(description="Parser for specific arguments for called function")
    specific_parser.add_argument('-v', '--video_path', type=str,
                                 help="Video Path where the video to be analyzed can be found at, "
                                      "starting at project root folder")
    specific_parser.add_argument('-d', '--detector', type=str,
                                 help="Detector Model, default is scrfd. "
                                      "Currently available Detectors: ".format(DETECTORS),
                                 default="scrfd")

    if function_name == 'detect_faces':
        specific_parser.add_argument('-t', '--target_path', type=str,
                                     help="Target Path, where the detected images are saved in. "
                                          "Defaults to None, if they shouldn't be saved",
                                     default=None)
        specific_parser.add_argument('-r', '--required_size', type=int,
                                     help="Patch size output (Tuple(int,int)), if specific size is required. "
                                          "Defaults to None, if patch size is neglectable",
                                     default=None)
    elif function_name == 'online_pipeline':
        specific_parser.add_argument('-c', '--classifier', type=str,
                                     help="Classification Model, default is one self-implemented classifier, "
                                          "that first embeds the patch and then measures the similarity. "
                                          "Currently available Classifiers (except meanshift): ".format(CLASSIFIERS),
                                     default="online")
    elif function_name == 'offline_pipeline':
        specific_parser.add_argument('-c', '--classifier', type=str,
                                     help="Classification Model, default is meanshift. "
                                          "Currently available Classifiers: ".format(CLASSIFIERS),
                                     default="meanshift")
    else:
        msg = "Error: Make sure you spelled the function type correctly. " \
              "Currently available functions: " + ", ".join(functions_dic.keys())
        print(msg)
        return None, None

    function = functions_dic[function_name]
    specific_args = specific_parser.parse_known_args()[0]
    return function, specific_args


if __name__ == '__main__':
    """The main function to call all functions in this Repo with
    """
    fun, args = parse_args()
    if fun is not None:
        fun(**vars(args))
