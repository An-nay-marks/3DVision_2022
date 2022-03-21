import os
import sys
import cv2
import face_recognition

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def extract_faces(video_path):
    inference_scale = 0.5
    assert (inference_scale <= 1)

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    file_name = os.path.basename(video_path)

    delay = int(1000 / fps)
    valid, frame = capture.read()

    while valid and cv2.waitKey(delay) & 0xFF != ord('q'):
        bgr_frame = cv2.resize(frame, (0, 0), fx=inference_scale, fy=inference_scale)[:, :, ::-1]
        face_locations = face_recognition.face_locations(bgr_frame)

        for i, face_bounds in enumerate(face_locations):
            top, right, bottom, left = [int(bound / inference_scale) for bound in face_bounds]
            name = f'Person {i+1}'
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
            cv2.putText(frame, name, (left, bottom + 25), font, 1.0, WHITE, 1)

        cv2.imshow(file_name, frame)
        valid, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()


def main():
    extract_faces(sys.argv[1])


if __name__ == '__main__':
    main()
