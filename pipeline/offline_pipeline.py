import warnings
import numpy as np

from pipeline.pipeline_utils import *
from utils_3DV import *


class OfflinePipeline:
    def __init__(self, source, run_name, export_size, detector=None, classifier=None, deca=None):
        self.source = source
        self.run_name = run_name
        self.export_size = export_size
        self.detector = detector
        self.classifier = classifier
        self.deca = deca
        self.target_dir = f"{OUT_DIR}/{self.run_name}"
        self.faces = []
        self.bboxes = None
        warnings.filterwarnings("ignore", category=UserWarning)
        
    def run(self):
        if not init_dir(self.target_dir):
            return
        if self.detector is None and self.classifier is None:
            print("Loading classified patches...")
            self.faces, self.identities = load_classified_patches(self.source)
        else:
            if self.detector is not None:
                self.detect()
                self.source.release()

            if self.classifier is None:
                return

            self.classify()
        
        if self.deca is None:
            self.save_classification()
        else:
            print("Reconstructing faces...")
            with tqdm(total=len(self.identities)) as pbar:
                for identity in np.unique(self.identities):
                    name = identity if self.classifier is None else self.classifier.get_name(identity)
                    sample_dir = create_id_export_dir(self.target_dir, name)
                    
                    id_patches = [self.faces[i] for i in range(len(self.identities)) if self.identities[i] == identity]

                    if self.deca is None:
                        for patch_idx, patch in enumerate(id_patches):
                            patch_name = f'patch_{patch_idx + 1}'
                            patch = resize_face(patch, self.export_size)
                            cv2.imwrite(os.path.join(sample_dir, f'{patch_name}.jpg'), patch)
                            pbar.update(1)
                    else:
                        reconstructions = self.deca.reconstruct_multiple(id_patches)
                        if len(reconstructions) == 1:
                            # no need for patch directory if there is only one reconstruction
                            self.deca.save_obj(os.path.join(sample_dir, f'{name}.obj'), reconstructions[0])
                        else:
                            for patch_idx, reconstruction in enumerate(reconstructions):
                                patch_name = f'patch_{patch_idx + 1}'
                                patch_dir = os.path.join(sample_dir, patch_name)
                                os.makedirs(patch_dir)
                                self.deca.save_obj(os.path.join(patch_dir, f'{patch_name}.obj'), reconstruction)

                        pbar.update(len(id_patches))
    
    def get_source(self):
        return int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def detect(self, notifier = None):
        print("Detecting faces...")
        num_frames = self.get_source()
        for frame_idx in tqdm(range(num_frames)):
            if frame_idx < 1000 or frame_idx > 1050:
                continue
            if notifier is not None:
                notifier.status(frame_idx)
            valid, frame = self.source.read()
            bboxes = self.detector.detect(frame)
            
            for face_idx, face in enumerate(bboxes):
                left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))

                if min(bottom - top, right - left) <= 110:
                    continue

                face_patch = frame[top:bottom + 1, left:right + 1]

                if self.classifier is None:
                    sample_dir = create_anonymous_export_dir(self.target_dir, frame_idx)
                    face_patch = resize_face(face_patch, self.export_size)
                    cv2.imwrite(os.path.join(sample_dir, f'patch_{face_idx + 1}.jpg'), face_patch)
                else:
                    self.faces.append(face_patch)

    def classify(self):
        print("Loading unclassified patches...")
        self.faces = load_raw_patches(self.source)
        print("Classifying faces...")
        self.faces = np.asarray(self.faces, dtype=object)
        self.identities, best_idx = self.classifier.classify_all(self.faces)
        self.faces = self.faces[best_idx]
        self.identities = self.identities[best_idx]


    def save_classification(self):
        print("Exporting patches...")
        with tqdm(total=len(self.identities)) as pbar:
            for identity in np.unique(self.identities):
                name = identity if self.classifier is None else self.classifier.get_name(identity)
                sample_dir = create_id_export_dir(self.target_dir, name)
                
                id_patches = [self.faces[i] for i in range(len(self.identities)) if self.identities[i] == identity]

                if self.deca is None:
                    for patch_idx, patch in enumerate(id_patches):
                        patch_name = f'patch_{patch_idx + 1}'
                        patch = resize_face(patch, self.export_size)
                        cv2.imwrite(os.path.join(sample_dir, f'{patch_name}.jpg'), patch)
                        pbar.update(1)
