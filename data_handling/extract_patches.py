import os
import sys
import time
import cv2
import onnxruntime
from utils import ROOT_DIR

from insightface.detection.scrfd.tools import scrfd


def analyze_video(video_path, target_path):
    video_path = f"{ROOT_DIR}/{video_path}"
    capture = cv2.VideoCapture(video_path)
    target = target_path
    if not os.path.exists(target):
        os.mkdir(target)
    image_id = 0
    valid, frame = capture.read()

    if not valid:
        raise RuntimeError(video_path + " is not a valid video file")

    file_name = os.path.basename(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = max(1.0, 1000 / fps)

    # width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    height, width, channels = frame.shape

    model_file = 'onnx/scrfd_34g.onnx'
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    so = onnxruntime.SessionOptions()
    so.inter_op_num_threads = 4
    so.intra_op_num_threads = 2
    session = onnxruntime.InferenceSession(model_file, so)
    # session = onnxruntime.InferenceSession(model_file, providers)
    detector = scrfd.SCRFD(model_file, session)
    detector.prepare(0, input_size=(640, 640))

    key = cv2.waitKey(1)
    t1 = time.time_ns()

    while valid and key & 0xFF != ord('q'):
        rgb_frame = frame[:, :, ::-1]
        bboxes, kpss = detector.detect(rgb_frame)

        for i, face_bounds in enumerate(bboxes):
            left, top, right, bottom, score = face_bounds.astype(int)
            
            # add 30% of padding everywhere, so we get the whole head instead of just the face
            padding_up_down = int(abs(top-bottom)*0.3)
            padding_up = min([padding_up_down, top])
            padding_down = min([padding_up_down, height-bottom])
            padding_left_right = int(abs(right-left)*0.3)
            padding_left = min([padding_left_right, left])
            padding_right = min([padding_left_right, width-right])
            name = f'Person {i+1}'
            font = cv2.FONT_HERSHEY_DUPLEX
            left -= padding_left
            top -= padding_up
            right += padding_right
            bottom += padding_down
            
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 25), font, 1.0, (255, 255, 255), 1)
            
            # save face patch
            file_name = f"{target_path}/{image_id}.png"
            image_id+=1
            cv2.imwrite(filename=file_name, img=frame[top:bottom,left:right,::])

        t2 = time.time_ns()
        processing_delay = (t2 - t1) / 1e6
        delay = max(1, round(frame_delay - processing_delay))
        key = cv2.waitKey(delay)

        t1 = time.time_ns()
        # cv2.imshow(file_name, frame)

        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()