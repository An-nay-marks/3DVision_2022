from .detect_faces import face_detection
from .encode_faces import encode_faces
from .classify_faces import classify_faces, classify_vgg
from .run_DECA import reconstruct
from utils_3DV import CLASSIFIERS

def run_offline_pipeline(video_path, detector = "scrfd", classifier="similarity", patches_dir = None, recon_dir=None):
    """sequentially runs each function in the pipeline at a time, which allows offline classification by clustering all extracted patches at once.

    Args:
        video_path (string): video path starting at ROOT directory
        detector (str, optional): The face detection model to use. Defaults to "scrfd".
        classifier (str, optional): The classifier to use. Defaults to "similarity".
        patches_dir (string, optional): If patches should be saved as images, specify target path starting at ROOT directory. Defaults to None, if detected face patches should not specifically be saved.
        recon_dir (string, optional): Specify wanted reconstruction directory to save DECA reconstructions in. If not specified, the datetime at runtime will be used as a folder name. Defaults to None.
    """
    if classifier not in CLASSIFIERS:
        raise RuntimeError(f"The given classifier is not valid. Valid classifiers are {CLASSIFIERS}")
    print("Starting with Face Detection...")
    faces = face_detection(video_path, patches_dir, detector)
    print("...Done")
    if classifier == "vgg":
        print("Running VGG")
        identities = classify_vgg(faces)
    else:
        print("Starting with Encoding faces...")
        faces, encodings = encode_faces(image_list=faces)
        print("...Done")
        #TODO: implement different classifiers, that can also be used for better offline classification
        print("Starting with Face Identification...")
        identities, identifier = classify_faces(encodings)
        print("...Done")
    '''
    print("Starting with DECA...")
    output_directory = reconstruct(faces, identities, identifier, target_dir=recon_dir)
    print(f"...Done!\nLook at folder {output_directory} to see the reconstructions.")
    if patches_dir is not None:
        print(f"Look at folder {patches_dir} to see the extracted face patches.")
    print("Goodbye! :)")'''