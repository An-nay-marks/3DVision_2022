import warnings
import numpy as np

from pipeline.pipeline_utils import *
from utils_3DV import *


def run(source, run_name, export_size, detector=None, classifier=None, deca=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    target_dir = f"{OUT_DIR}/{run_name}"
    # logs_dir = f"{LOGS_DIR}/{run_name}"

    if not init_dir(target_dir):
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
                if frame_idx < 10000 or frame_idx > 10400:
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

    print("Exporting patches..." if deca is None else "Reconstructing faces...")

    with tqdm(total=len(identities)) as pbar:
        for identity in np.unique(identities):
            name = identity if classifier is None else classifier.get_name(identity)
            if str(identity) != "id_2":
                continue
            sample_dir = create_id_export_dir(target_dir, name)
            
            id_patches = [faces[i] for i in range(len(identities)) if identities[i] == identity]

            if deca is None:
                for patch_idx, patch in enumerate(id_patches):
                    patch_name = f'patch_{patch_idx + 1}'
                    patch = resize_face(patch, export_size)
                    cv2.imwrite(os.path.join(sample_dir, f'{patch_name}.jpg'), patch)
                    pbar.update(1)
            else:
                for patch_idx, patch in enumerate(id_patches):
                    reconstruction = deca.reconstruct(patch)
                    patch_name = f'patch_{patch_idx + 1}'
                    patch_dir = os.path.join(sample_dir, patch_name)
                    os.makedirs(patch_dir)
                    deca.save_obj(os.path.join(patch_dir, f'{patch_name}.obj'), reconstruction)
                    pbar.update(len(id_patches))
