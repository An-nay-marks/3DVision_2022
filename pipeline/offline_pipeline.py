from enum import unique
from turtle import shape
import numpy as np

from pipeline.pipeline_utils import *
from utils_3DV import *
import warnings

def run(source, run_name, export_size, detector=None, classifier=None, deca=None, optimize=None):
    warnings.filterwarnings("ignore", category=UserWarning) 
    if not init_dir(run_name):
        return
    target_dir = f"{OUT_DIR}/{run_name}"
    logs_dir = f"{LOGS_DIR}/{run_name}"
    if optimize is not None and deca is None:
        print("Error: Optimizer can only be called if DECA is selected!")
        return
    if detector is None and classifier is None:
        print("Loading classified patches...")
        faces, identities = load_classified_patches(source)
    else:
        if detector is None:
            print("Loading unclassified patches...")
            faces = load_raw_patches(source)
        else:
            print("Detecting faces...")
            faces = []
            num_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in tqdm(range(num_frames)):
                #if frame_idx > 1000:
                #    break
                valid, frame = source.read()
                bboxes = detector.detect(frame)

                for face_idx, face in enumerate(bboxes):
                    left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))

                    if min(bottom - top, right - left) <= 110:
                        continue

                    face_patch = frame[top:bottom + 1, left:right + 1]

                    if classifier is None:
                        sample_dir = create_anonymous_export_dir(target_dir, frame_idx)
                        face_patch = resize_face(face_patch, export_size)
                        cv2.imwrite(os.path.join(sample_dir, f'patch_{face_idx + 1}.jpg'), face_patch)
                    else:
                        faces.append(face_patch)

            source.release()

        if classifier is None:
            return

        print("Classifying faces...")
        faces = np.asarray(faces, dtype=object)
        identities, best_idx = classifier.classify_all(faces)

        faces = faces[best_idx]
        identities = identities[best_idx]
    
    if deca is None:
        print("Exporting patches...")
        for face_idx, (identity, patch) in enumerate(tqdm(zip(identities, faces), total=len(faces))):
            name = identity if classifier is None else classifier.get_name(identity)
            sample_dir = create_id_export_dir(target_dir, name)
            patch = resize_face(patch, export_size)
            cv2.imwrite(os.path.join(sample_dir, f'patch_{face_idx + 1}.jpg'), patch)
        
    else:
        # Reconstruction, optionally with optimizer
        if optimize == "mean":
            print("Reconstructing faces by averaging all face parameters...")
            ids = np.unique(identities)
            for unique_id in tqdm(ids):
                # filter for unique id, and average all faces from the corresponding id
                unique_id_broadcasted = np.full(fill_value=unique_id, shape=len(identities))
                face_mask = np.where(identities == unique_id_broadcasted, True, False)
                unique_id_faces = np.asarray(faces)[face_mask].tolist()
                reconstruction, _ = deca.decode(deca.encode_average(unique_id_faces))
                name = f'{unique_id}_mean'
                id_dir = create_id_export_dir(target_dir, name)
                os.makedirs(id_dir, exist_ok=True)
                path = os.path.join(id_dir, f'{name}.obj')
                deca.save_obj(path, reconstruction)
        
        elif optimize == "mean_shape":
            print("Averaging shape parameters for reconstruction...")
            unique_ids = np.unique(identities)
            average_shape_parameters = []
            for unique_id in tqdm(unique_ids):
                # filter for unique id, and average all faces from the corresponding id
                unique_id_broadcasted = np.full(fill_value=unique_id, shape=len(identities))
                face_mask = np.where(identities == unique_id_broadcasted, True, False)
                unique_id_faces = np.asarray(faces)[face_mask].tolist()
                average_shape = deca.get_average_shape_param(unique_id_faces)
                average_shape_parameters.append(average_shape)
            print("Reconstructing with average shape parameters...")
            for face_idx, (identity, patch) in enumerate(tqdm(zip(identities, faces), total=len(faces))):
                name = identity if classifier is None else classifier.get_name(identity)
                sample_dir = create_id_export_dir(target_dir, name)
                average_shape_parameter_index = np.where(unique_ids==identity)
                average_shape = average_shape_parameters[average_shape_parameter_index[0].item()]
                reconstruction, _ = deca.decode(deca.encode(patch, average_shape))
                sample_name = f'patch_{face_idx + 1}'
                sample_dir = os.path.join(sample_dir, sample_name)
                os.makedirs(sample_dir, exist_ok=True)
                path = os.path.join(sample_dir, f'{sample_name}.obj')
                deca.save_obj(path, reconstruction)
        
        else:
            print("Reconstructing faces...")
            for face_idx, (identity, patch) in enumerate(tqdm(zip(identities, faces), total=len(faces))):
                name = identity if classifier is None else classifier.get_name(identity)
                sample_dir = create_id_export_dir(target_dir, name)        
                reconstruction, _ = deca.decode(deca.encode(patch))
                sample_name = f'patch_{face_idx + 1}'
                sample_dir = os.path.join(sample_dir, sample_name)
                os.makedirs(sample_dir, exist_ok=True)
                path = os.path.join(sample_dir, f'{sample_name}.obj')
                deca.save_obj(path, reconstruction)