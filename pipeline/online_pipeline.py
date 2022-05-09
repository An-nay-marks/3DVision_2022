from pipeline.pipeline_utils import *
from utils_3DV import *
import warnings


def run(provider, run_name, export_size, detector, classifier=None, deca=None):
    warnings.filterwarnings("ignore", category=UserWarning) 
    if not init_dir(run_name):
        return
    target_dir = f"{OUT_DIR}/{run_name}"
    # logs_dir = f"{LOGS_DIR}/{run_name}"
    valid, frame = provider.read()

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

        for face_idx, face in enumerate(bboxes):
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

        valid, frame = provider.read()

    provider.release()
    cv2.destroyAllWindows()
