import os
import sys
import cv2
import time

from insightface.detection.scrfd.tools import scrfd

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def analyze_video(video_path):
    onnx_path = 'onnx/scrfd_34g.onnx'
    detector = scrfd.SCRFD(model_file=onnx_path)

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    file_name = os.path.basename(video_path)

    # width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector.prepare(0, input_size=(640, 640))
    frame_delay = max(1.0, 1000 / fps)

    valid, frame = capture.read()
    key = cv2.waitKey(1)
    t1 = time.time_ns()

    while valid and key & 0xFF != ord('q'):
        rgb_frame = frame[:, :, ::-1]
        bboxes, kpss = detector.detect(rgb_frame)

        for i, face_bounds in enumerate(bboxes):
            left, top, right, bottom, score = face_bounds.astype(int)
            name = f'Person {i+1}'
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
            cv2.putText(frame, name, (left, bottom + 25), font, 1.0, WHITE, 1)

        t2 = time.time_ns()
        processing_delay = (t2 - t1) / 1e6
        delay = max(1, round(frame_delay - processing_delay))
        key = cv2.waitKey(delay)
        cv2.imshow(file_name, frame)
        t1 = time.time_ns()

        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    analyze_video(sys.argv[1])


if __name__ == '__main__':
    main()
