import os
import sys
import time
import cv2
import onnxruntime

from insightface.detection.scrfd.tools import scrfd


def analyze_video(video_path):
    capture = cv2.VideoCapture(video_path)
    valid, frame = capture.read()

    if not valid:
        raise RuntimeError(video_path + " is not a valid video file")

    file_name = os.path.basename(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = max(1.0, 1000 / fps)

    # width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_file = 'onnx/scrfd_34g.onnx'
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'])
    session = onnxruntime.InferenceSession(model_file, None, providers)
    detector = scrfd.SCRFD(model_file, session)
    detector.prepare(0, input_size=(640, 640))

    key = cv2.waitKey(1)
    t1 = time.time_ns()

    while valid and key & 0xFF != ord('q'):
        rgb_frame = frame[:, :, ::-1]
        bboxes, kpss = detector.detect(rgb_frame)

        for i, face_bounds in enumerate(bboxes):
            left, top, right, bottom, score = face_bounds.astype(int)
            name = f'Person {i+1}'
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 25), font, 1.0, (255, 255, 255), 1)

        t2 = time.time_ns()
        processing_delay = (t2 - t1) / 1e6
        delay = max(1, round(frame_delay - processing_delay))
        key = cv2.waitKey(delay)

        t1 = time.time_ns()
        cv2.imshow(file_name, frame)

        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()