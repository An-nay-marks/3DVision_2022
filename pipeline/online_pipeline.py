import time

from pipeline.pipeline_utils import *
from utils_3DV import init_dir
import warnings
import numpy as np


def run(provider, target_dir, export_size, detector, classifier=None, deca=None):
    warnings.filterwarnings("ignore", category=UserWarning) 
    if not init_dir(target_dir):
        return

    valid, frame = provider.read()
    frame_flipped = cv2.flip(frame, 1)

    frame_number = 0

    #count the total of detected faces in the original and flipped versions. also count matches
    total_original = 0
    total_flipped = 0
    total_matches = 0
    #find all valid bboxes (ones that match in the original and flipped version)
    valid_bboxes = []
    frame_count = 0 #only to keep track of valid_bboxes

    if not valid:
        raise RuntimeError("Not a valid video source")

    fps = provider.get(cv2.CAP_PROP_FPS)
    frame_time = max(1.0, 1000 / fps)
    frame_idx = -1

    key = cv2.waitKey(1)
    t1 = time.time_ns()
    
    while valid and key & 0xFF != ord('q'):
        frame_idx += 1
        bboxes = detector.detect(frame)
        bboxes_flipped = detector.detect(frame_flipped)

        #initialize an array that keeps the indices of the matches between the original and flipped versions
        matches = np.full(bboxes.shape[0], -1)

        if len(matches) == 0: #no face detected in original frame at all, go to next frame
            frame_number += 1
            valid, frame = provider.read()
            frame_flipped = cv2.flip(frame, 1)
            print('here')
            continue

        frame_count += 1 #only to keep track of valid_bboxes
        valid_bboxes.append(bboxes)

        #array containing all the matches, and the total amount of flipped frames
        matches, total_flipped = find_matches(bboxes, bboxes_flipped, matches)

        for face_idx, face in enumerate(bboxes):
            #we are in the loop, so a face is certainly detected in the original version
            total_original += 1

            #a face is detected in the original frame but there is no match in the flipped version
            if matches[face_idx] == -1:
                valid_bboxes[frame_count-1][face_idx] = 0
                continue

            #here we know the current face/bounding box is also detected in the flipped version, so we have a match!
            total_matches += 1
            
            left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if min(bottom-top, right-left) <= 110:
                continue

            face_patch = frame[top:bottom+1, left:right+1]

            if classifier is None:
                sample_dir = create_anonymous_export_dir(target_dir, frame_idx)
                face_patch = resize_face(face_patch, export_size)
                cv2.imwrite(os.path.join(sample_dir, f'patch_{face_idx + 1}.jpg'), face_patch)
            else:
                identity = classifier.classify(face_patch)
                name = classifier.get_name(identity)

                cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                sample_dir = create_id_export_dir(target_dir, name)
                sample_name = f'patch_{classifier.get_num_samples(identity)}'

                if deca is None:
                    face_patch = resize_face(face_patch, export_size)
                    cv2.imwrite(os.path.join(sample_dir, f'patch_{face_idx + 1}.jpg'), face_patch)
                else:
                    reconstruction, _ = deca.decode(deca.encode(face_patch))
                    sample_dir = os.path.join(sample_dir, sample_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    deca.save_obj(os.path.join(sample_dir, f'{sample_name}.obj'), reconstruction)

        t2 = time.time_ns()
        processing_time = (t2 - t1) / 1e6
        delay = max(1, round(frame_time - processing_time))
        key = cv2.waitKey(delay)

        t1 = time.time_ns()
        cv2.imshow('Video', frame)

        frame_number += 1
        valid, frame = provider.read()
        frame_flipped = cv2.flip(frame, 1)

    provider.release()
    cv2.destroyAllWindows()
