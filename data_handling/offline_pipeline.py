from .detect_faces import face_detection
from .encode_faces import encode_faces
from .classify_faces import classify_faces_online, classify_vgg, classify_meanshift, getNN
from .run_DECA import reconstruct
from utils_3DV import *
import os

def run_offline_pipeline(video_path, target_dir = None, detector = "scrfd", classifier="meanshift", patches_dir = None, recon_dir=None):
    """sequentially runs each function in the pipeline at a time, which allows offline classification by clustering all extracted patches at once.

    Args:
        video_path (str): video path starting at ROOT directory
        target_dir (str, optonal): target_dir starting at ROOT directory. If defaultes to None, the current datetime will be used to save reconstructions in the folder data/reconstructions
        detector (str, optional): The face detection model to use. Defaults to "scrfd".
        classifier (str, optional): The classifier to use. Defaults to "meanshift".
        patches_dir (str, optional): If patches should be saved as images, specify target path starting at ROOT directory. Defaults to None, if detected face patches should not specifically be saved.
        recon_dir (str, optional): Specify wanted reconstruction directory to save DECA reconstructions in. If not specified, the datetime at runtime will be used as a folder name. Defaults to None.
    """
    if classifier not in CLASSIFIERS:
        raise RuntimeError(f"The given classifier is not valid. Valid classifiers are {CLASSIFIERS}")
    
    if target_dir is None:
        target_dir = f"{ROOT_DIR}/data/reconstructions/{get_current_datetime_as_str()}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
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
        print("Starting with Face Identification...")
        if classifier == "online":
            identities, identifier = classify_faces_online(encodings)
        elif classifier == "meanshift":
            unique_labels, cluster_centers, identities = classify_meanshift(encodings)
        print("...Done")
    
    print("Starting with DECA...")
    if classifier == "meanshift":
        # reconstruct only neirest encodings to each cluster center:
        idxs = getNN(cluster_centers, encodings)
        faces = [faces[i] for i in idxs]
        identities = [identities[i] for i in idxs]
        recons, deca_model = reconstruct(faces)
    else:
        recons, deca_model = reconstruct(faces)
    print("...Done")
    for idx, reconstruction in enumerate(recons):
        if classifier == "online":
            person_reconstruction_number = identifier.identities[identity].num_encodings
            identity = identifier.get_identity(encodings[idx])
            name = f'{identity + 1}'
        else:
            # get number of folders in person_specific directory
            name = identities[idx]
            obj_dir = os.path.join(target_dir, f'id_{name}')
            os.makedirs(obj_dir, exist_ok=True)
            person_reconstruction_number=count_folders(obj_dir)
        obj_name = f'patch_{person_reconstruction_number}'
        obj_dir = os.path.join(target_dir, f'id_{name}', obj_name)
        os.makedirs(obj_dir, exist_ok=True)
        deca_model.save_obj(os.path.join(obj_dir, f'{obj_name}.obj'), reconstruction)
    print(f"...Done!\nLook at folder {target_dir} to see the reconstructions.")
    if patches_dir is not None:
        print(f"Look at folder {patches_dir} to see the extracted face patches.")
    print("Goodbye! :)")