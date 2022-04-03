from utils import ROOT_DIR
from reconstruction.deca import DECAReconstruction

def reconstruct(images, identities, identifier):
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'
    deca = DECAReconstruction(deca_file, flame_file, albedo_file)
    
    for idx, face_patch in enumerate(images):
        op_dict = deca.reconstruct(face_patch)
        identity = identities[idx]
        face_nr = identifier.identities[identity].num_encodings
        obj_name = f'patch_{face_nr}'
        obj_dir = os.path.join(out_directory, f'id_{identity+1}', obj_name)
        os.makedirs(obj_dir, exist_ok=True)
        deca.save_obj(os.path.join(obj_dir, f'{obj_name}.obj'), op_dict)