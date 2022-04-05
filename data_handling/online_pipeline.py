import os
import time
import cv2
import shutil
import time
import cv2
from utils_3DV import *
from detection import scrfd, yolo5
from recognition import arcface, face_identifier, vgg
from reconstruction.deca import DECAReconstruction
from data_handling.detect_faces import pad_face

def run_online_pipeline(video_path, target_dir = None, detector = "scrfd", classifier = "online"):
    """One loop of the pipeline immediatly for each frame. Classification Types are therefore limited

    Args:
        video_path (str): video path starting at ROOT directory
        target_dir (str, optional): target_dir starting at ROOT directory. If defaultes to None, the current datetime will be used to save reconstructions in the folder data/reconstructions
        detector (str, optional): The face detection model to use. Defaults to "scrfd".
        classifier (str, optional): The classifier to use. Defaults to "online".

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_
    """
    # check all prerequisites in the beginning
    if detector not in DETECTORS:
        raise RuntimeError(f"{detector} is not a valid face detector")
    if classifier not in CLASSIFIERS or classifier == "meanshift":
        raise RuntimeError(f"{classifier} is not a valid face classifier for the online pipeline.")
    if target_dir is None:
        target_dir = f"{ROOT_DIR}/data/reconstructions/{get_current_datetime_as_str()}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    capture = cv2.VideoCapture(video_path)
    valid, frame = capture.read()
    if not valid:
        raise RuntimeError(video_path + " is not a valid video file")

    file_name = os.path.basename(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_time = max(1.0, 1000 / fps)

    detector = scrfd.SCRFaceDetector(f'{ROOT_DIR}/data/model_files/scrfd_34g.onnx') if detector=="scrfd" else yolo5.YOLOv5FaceDetector(f'{ROOT_DIR}/data/model_files/yolov5l.pt')
    if classifier == "online":
        encoder = arcface.ArcFaceR100(f'{ROOT_DIR}/data/model_files/arcface_r100.pth')
        identifier = face_identifier.RealTimeFaceIdentifier(threshold=0.3)
    elif classifier == "vgg":
        identifier = vgg.VGGEncoderAndClassifier(threshold=0.3)
    
    # deca currently not supported for vgg
    deca_file = f'{ROOT_DIR}/data/model_files/deca_model.tar'
    flame_file = f'{ROOT_DIR}/data/model_files/generic_model.pkl'
    albedo_file = f'{ROOT_DIR}/data/model_files/FLAME_albedo_from_BFM.npz'
    deca = DECAReconstruction(deca_file, flame_file, albedo_file)

    key = cv2.waitKey(1)
    t1 = time.time_ns()
    
    oo = 0
    while valid and key & 0xFF != ord('q'):
        # use to skip some frames
        '''if oo < 80 and oo > 200:
            t2 = time.time_ns()
            processing_time = (t2 - t1) / 1e6
            delay = max(1, round(frame_time - processing_time))
            key = cv2.waitKey(delay)
            valid, frame = capture.read()
            continue'''
        bboxes = detector.detect(frame)

        for i, face in enumerate(bboxes):
            left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))
            identity = -1
            name = "not recognized"

            if min(bottom-top, right-left) > 110:
                face_patch = frame[top:bottom+1, left:right+1]
                if classifier == "online":
                    encoding = encoder.encode(face_patch)
                    identity = identifier.get_identity(encoding)
                    name = f'{identity + 1}'
                elif classifier == "vgg":
                    identity = identifier.classify(face_patch)
                    if identity[0]!= -1:
                        name = identity[0]
                    else:
                        name = "Unknown"
                op_dict = deca.reconstruct(face_patch)
                # save
                if classifier == "online":
                    person_reconstruction_number = identifier.identities[identity].num_encodings
                else:
                    # get number of folders in person_specific directory
                    obj_dir = os.path.join(target_dir, f'id_{name}')
                    os.makedirs(obj_dir, exist_ok=True)
                    person_reconstruction_number=count_folders(obj_dir)
                obj_name = f'patch_{person_reconstruction_number}'
                obj_dir = os.path.join(target_dir, f'id_{name}', obj_name)
                os.makedirs(obj_dir, exist_ok=True)
                deca.save_obj(os.path.join(obj_dir, f'{obj_name}.obj'), op_dict)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if identity != -1:
                cv2.putText(frame, name, (left, bottom + 25), font, 1.0, (255, 255, 255), 1)

        t2 = time.time_ns()
        processing_time = (t2 - t1) / 1e6
        delay = max(1, round(frame_time - processing_time))
        key = cv2.waitKey(delay)

        t1 = time.time_ns()
        cv2.imshow(file_name, frame)

        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()