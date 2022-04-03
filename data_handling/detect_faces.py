import os
from unittest.mock import patch
import onnxruntime
import time
import cv2
from utils_3DV import ROOT_DIR, DETECTORS

from detection import scrfd, yolo5


def face_detection(video_path, target_path=None, detector="scrfd"):
    """Extract patches, detected as faces from the video. If target_path is not specified, patches will not be saved, but only returned.

    Args:
        video_path (string): Video Path starting from ROOT directory
        target_path (string, optional): Video Path starting from ROOT directory. If nto specified, patches will not be saved as images.
        detector (str, optional): the face detector model to use. Defaults to "scrfd".

    Raises:
        RuntimeError: See error message

    Returns:
        list: list of extracted patches
    """
    if detector not in DETECTORS:
        raise RuntimeError(f"{detector} is not a valid face detector")
    video_path = f"{ROOT_DIR}/{video_path}"
    capture = cv2.VideoCapture(video_path)
    target = target_path
    if not os.path.exists(target) and target is not None:
        os.mkdir(target)
    image_id = 0
    valid, frame = capture.read()

    if not valid:
        raise RuntimeError(video_path + " is not a valid video file")

    file_name = os.path.basename(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = max(1.0, 1000 / fps)

    detector = scrfd.SCRFaceDetector(f'{ROOT_DIR}/data/model_files/scrfd_34g.onnx') if detector=="scrfd" else yolo5.YOLOv5FaceDetector(f'{ROOT_DIR}/data/model_files/yolov5l.pt')
    ''' providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'])
    so = onnxruntime.SessionOptions()
    so.inter_op_num_threads = 4
    so.intra_op_num_threads = 2
    session = onnxruntime.InferenceSession(model_file, so)
    # session = onnxruntime.InferenceSession(model_file, providers)
    detector = scrfd.SCRFD(model_file, session)
    detector.prepare(0, input_size=(640, 640))'''

    key = cv2.waitKey(1)
    t1 = time.time_ns()
    images = []

    while valid and key & 0xFF != ord('q'):
        bboxes = detector.detect(frame)

        for i, face in enumerate(bboxes):
            left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))
            name = f'Person {i+1}'
            font = cv2.FONT_HERSHEY_DUPLEX
            if min(bottom-top, right-left) > 110:
                face_patch = frame[top:bottom+1, left:right+1]
                name = f'Person {i+1}'
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 25), font, 1.0, (255, 255, 255), 1)
                if min(bottom-top, right-left) > 110:
                    if target_path is not None:
                        # save face patch
                        file_name = f"{target_path}/{image_id}.png"
                        image_id+=1
                        cv2.imwrite(filename=file_name, img=face_patch)
                    images.append(face_patch)

        t2 = time.time_ns()
        processing_delay = (t2 - t1) / 1e6
        delay = max(1, round(frame_delay - processing_delay))
        key = cv2.waitKey(delay)

        t1 = time.time_ns()
        # cv2.imshow(file_name, frame)

        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()
    return images
    
def pad_face(img, left, top, right, bottom):
    factor = 0.35
    pad_x = round((right - left) * factor / 2)
    pad_y = round((bottom - top) * factor / 2)

    left = max(0, left - pad_x)
    right = min(img.shape[1], right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(img.shape[0], bottom + pad_y)
    return left, top, right, bottom