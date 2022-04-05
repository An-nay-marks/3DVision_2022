from utils_3DV import ROOT_DIR, get_current_datetime_as_str
from reconstruction.deca import DECAReconstruction
import os
import shutil

def reconstruct(images):
    """Runs DECA reconstructions for all images.

    Args:
        images (list): list of facial images to reconstruct
        identities (list): list of identities in order of images
        identifier (Identification Model): the classifier used to create the identities
        target_dir (string, optional): Target Directory starting at ROOT directory to save the reconstructions in. If not specified, it uses the datetime at runtime to create the target directory. Defaults to None.

    Returns:
        string: The global target directory, where the reconstructions can be found in.
    """    
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'
    deca = DECAReconstruction(deca_file, flame_file, albedo_file)
    recons = []
    for face_patch in images:
        recons.append(deca.reconstruct(face_patch))
    return recons, deca