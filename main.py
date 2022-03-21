import os
import sys
import cv2

from insightface.detection.scrfd.tools import scrfd

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def analyze_video(video_path):
    # onnx_path = './insightface/detection/scrfd/onnx/scrfd_34g_shape1920x1080.onnx'
    onnx_path = './insightface/detection/scrfd/onnx/scrfd_34g.onnx'
    detector = scrfd.SCRFD(model_file=onnx_path)

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    file_name = os.path.basename(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector.prepare(-1)

    delay = int(1000 / fps)
    valid, frame = capture.read()

    while valid and cv2.waitKey(delay) & 0xFF != ord('q'):
        rgb_frame = frame[:, :, ::-1]
        bboxes, kpss = detector.detect(rgb_frame, input_size=(640, 640))

        for i, face_bounds in enumerate(bboxes):
            # score = face_bounds[4]
            left, top, right, bottom, score = face_bounds.astype(int)

            name = f'Person {i+1}'
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
            cv2.putText(frame, name, (left, bottom + 25), font, 1.0, WHITE, 1)

        cv2.imshow(file_name, frame)
        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()


def main():
    analyze_video(sys.argv[1])


if __name__ == '__main__':
    main()
