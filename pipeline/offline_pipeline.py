import numpy as np

from pipeline.pipeline_utils import *
from utils_3DV import *
import warnings

def run(source, run_name, export_size, detector=None, classifier=None, deca=None):
    warnings.filterwarnings("ignore", category=UserWarning) 
    if not init_dir(run_name):
        return
    target_dir = f"{OUT_DIR}/{run_name}"
    logs_dir = f"{LOGS_DIR}/{run_name}"
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
                if frame_idx > 500:
                    break
                elif frame_idx < 400:
                    valid, frame = source.read()
                    continue
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
        reconstructions, identity_names, sample_names = deca.reconstruct(identities, faces, classifier)
        # save reconstructions by identity and optionally sample name
        use_sample_names = len(sample_names) > 0
        print("Exporting reconstructions...")
        for idx, (reconstruction, identity_name) in enumerate(tqdm(zip(reconstructions, identity_names), total=len(reconstructions))):
            target_subdir = create_id_export_dir(target_dir, identity_name)
            object_name = identity_name
            if use_sample_names:
                sample_name = sample_names[idx]
                target_subdir = os.path.join(target_subdir, sample_name)
                os.makedirs(target_subdir, exist_ok=True)
                object_name = sample_name
            path = os.path.join(target_subdir, f'{object_name}.obj')
            deca.save_obj(path, reconstruction)