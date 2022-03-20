import os
import sys
import cv2
import face_recognition

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def extract_faces(video_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    file_name = os.path.basename(video_path)

    min_width = 50
    min_height = 50

    delay = int(1000 / fps)
    valid, frame = capture.read()

    inv_scale = 4
    scale = 1.0 / inv_scale

    while valid and cv2.waitKey(delay) & 0xFF != ord('q'):
        bgr_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)[:, :, ::-1]
        face_locations = face_recognition.face_locations(bgr_frame)

        for i, face_bounds in enumerate(face_locations):
            top, right, bottom, left = [inv_scale * bound for bound in face_bounds]
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
