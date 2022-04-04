import argparse
import sys
from data_handling.detect_faces import face_detection
from data_handling.online_pipeline import run_online_pipeline
from data_handling.offline_pipeline import run_offline_pipeline
from utils_3DV import DETECTORS

if __name__ == '__main__':
    """The main function to call all functions in this Repo with
    """
    
    # dict of functions that can be called
    functions_dic = {'detect_faces': face_detection,
                     'online_pipeline': run_online_pipeline,
                     'offline_pipeline': run_offline_pipeline}
    
    # parent parser
    parser = argparse.ArgumentParser(description="Implementation of the 3D Face Reconstruction Pipeline using Videos", add_help=False)
    parser.add_argument('-f', '--function', default="detect_faces", type=str, help="Give the function you would like to run. Default = \"detect_faces\".\nCurrently available:\n".join(functions_dic.keys()))

    # specific parser to each function
    specific_parser = argparse.ArgumentParser(description = "Parser for specific arguments for called function")
    args = parser.parse_known_args()[0]
    if args.function.lower() == 'detect_faces':
        specific_parser.add_argument('-v', '--video_path', type=str, help="Video Path where the video to be analyzed can be found at, starting at project root folder")
        specific_parser.add_argument('-t', '--target_path', type=str, help="Target Path, where the detected images are saved in. Defaults to None, if they shouldn't be saved", default=None)
        specific_parser.add_argument('-d', '--detector', type=str, help="Detector Model, default is scrfd. Currently available Detectors: ".format(DETECTORS), default="scrfd")
        specific_parser.add_argument('-r', '--required_size', type=int, help="Patch size output (Tuple(int,int)), if specific size is required. Defaults to None, if patch size is neglectable", default=None)
        filter_args = ["video_path", "target_path", "detector", "required_size"]
    elif args.function.lower() == 'online_pipeline':
        specific_parser.add_argument('-v', '--video_path', type=str, help="Video Path where the video to be analyzed can be found at, starting at project root folder")
        specific_parser.add_argument('-d', '--detector', type=str, help="Detector Model, default is scrfd. Currently available Detectors: ".format(DETECTORS), default="scrfd")
        filter_args = ["video_path", "detector"]
    elif args.function.lower() == 'offline_pipeline':
        specific_parser.add_argument('-v', '--video_path', type=str, help="Video Path where the video to be analyzed can be found at, starting at project root folder")
        specific_parser.add_argument('-d', '--detector', type=str, help="Detector Model, default is scrfd. Currently available Detectors: ".format(DETECTORS), default="scrfd")
        filter_args = ["video_path", "detector"]
    else:
        msg = "Error: Make sure you spelled the function type correctly. Currently available functions: " + ", ".join(functions_dic.keys())
        print(msg)        
        sys.exit(0)
    
    # call function with function-specific arguments
    try:
        spec_args = specific_parser.parse_known_args()[0]
        fun = functions_dic[args.function.lower()]
        fun(**{k: v for k, v in spec_args._get_kwargs()})
    except (KeyError, AttributeError) as err:
        print(err)
        print("Error: Make sure the parameters for "+ args.function + " are correct. They have to be contained in this list: " + ", ".join(filter_args))
