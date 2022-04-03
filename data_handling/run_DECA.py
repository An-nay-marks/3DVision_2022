from utils_3DV import ROOT_DIR, get_current_datetime_as_str
from reconstruction.deca import DECAReconstruction
import os
import shutil

def reconstruct(images, identities, identifier, target_dir=None):
    """Runs DECA reconstructions for all images.

    Args:
        images (list): list of facial images to reconstruct
        identities (list): list of identities in order of images
        identifier (Identification Model): the classifier used to create the identities
        target_dir (string, optional): Target Directory starting at ROOT directory to save the reconstructions in. If not specified, it uses the datetime at runtime to create the target directory. Defaults to None.

    Returns:
        string: The global target directory, where the reconstructions can be found in.
    """
    if target_dir is None:
        target_dir = f"{ROOT_DIR}/data/reconstructions/{get_current_datetime_as_str()}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'
    deca = DECAReconstruction(deca_file, flame_file, albedo_file)
    
    for idx, face_patch in enumerate(images):
        op_dict = deca.reconstruct(face_patch)
        identity = identities[idx]
        face_nr = identifier.identities[identity].num_encodings
        obj_name = f'patch_{face_nr}'
        obj_dir = os.path.join(target_dir, f'id_{identity+1}', obj_name)
        os.makedirs(obj_dir, exist_ok=True)
        deca.save_obj(os.path.join(obj_dir, f'{obj_name}.obj'), op_dict)
    
    return target_dir