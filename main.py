import argparse
import os


from detection import scrfd, yolo5
from recognition import arcface, face_identifier
from reconstruction.deca import DECAReconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="3D face reconstruction pipeline from video", add_help=False)
    parser.add_argument('-p', '--path', type=str)
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    """The main function to call all functions in this Repo with
    """
    
    # dict of functions that can be called
    functions_dic = {'detect_faces': ep.analyze_video}
    
    # parent parser
    parser = argparse.ArgumentParser(description="Implementation of the 3D Face Reconstruction Pipeline using Videos", add_help=False)
    parser.add_argument('-f', '--function', default="detect_faces", type=str, help="Give the function you would like to run. Default = \"detect_faces\".\nCurrently available:\n".join(functions_dic.keys()))

    # specific parser to each function
    specific_parser = argparse.ArgumentParser(description = "Parser for specific arguments for called function")
    args = parser.parse_known_args()[0]
    if args.function.lower() == 'detect_faces':
        specific_parser.add_argument('-v', '--video_path', type=str, help="Video Path where the video to be analyzed can be found at, starting at project root folder is enough")
        specific_parser.add_argument('-t', '--target_path', type=str, help="Target Path, where the detected images are saved in.")
        filter_args = ["video_path", "target_path"]
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
    args = parse_args()
    analyze_video(args.path)
