import argparse
import os
import time
import cv2

from detection import scrfd, yolo5


def parse_args():
    parser = argparse.ArgumentParser(description="3D face reconstruction pipeline from video", add_help=False)
    parser.add_argument('-v', '--video_path', type=str)
    return parser.parse_known_args()[0]


def analyze_video(video_path):
    capture = cv2.VideoCapture(video_path)
    valid, frame = capture.read()

    if not valid:
        raise RuntimeError(video_path + " is not a valid video file")

    file_name = os.path.basename(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = max(1.0, 1000 / fps)

    # detector = scrfd.SCRFaceDetector('model_files/scrfd_34g.onnx')
    detector = yolo5.YOLOv5FaceDetector('model_files/yolov5l.pt')

    key = cv2.waitKey(1)
    t1 = time.time_ns()

    while valid and key & 0xFF != ord('q'):
        bboxes = detector.detect(frame)

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


if __name__ == '__main__':
    args = parse_args()
    analyze_video(args.video_path)
