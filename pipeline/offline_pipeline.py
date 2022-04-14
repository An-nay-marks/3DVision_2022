import numpy as np
from tqdm import tqdm

from pipeline.pipeline_utils import *
from utils_3DV import *


def run(source, target_dir, detector, classifier=None, deca=None):
    init_dir(target_dir)

    if detector is None and classifier is None:
        faces, identities = load_classified_patches(source)
    else:
        if detector is None:
            faces = load_raw_patches(source)
        else:
            print("Detecting faces...")
            faces = []
            num_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in tqdm(range(num_frames)):
                valid, frame = source.read()
                bboxes = detector.detect(frame)

                for face_idx, face in enumerate(bboxes):
                    left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))

                    if min(bottom - top, right - left) <= 110:
                        continue

                    face_patch = frame[top:bottom + 1, left:right + 1]

                    if classifier is None:
                        sample_dir = create_anonymous_export_dir(target_dir, frame_idx)
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
    for face_idx, (identity, patch) in enumerate(tqdm(zip(identities, faces), total=len(faces))):
        name = identity if classifier is None else classifier.get_name(identity)
        sample_dir = create_id_export_dir(target_dir, name)

        if deca is None:
            cv2.imwrite(os.path.join(sample_dir, f'patch_{face_idx + 1}.jpg'), patch)
        else:
            reconstruction = deca.reconstruct(patch)
            sample_name = f'patch_{face_idx + 1}'
            sample_dir = os.path.join(sample_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)
            deca.save_obj(os.path.join(sample_dir, f'{sample_name}.obj'), reconstruction)
