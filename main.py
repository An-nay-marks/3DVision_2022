import argparse
import os
import shutil
import time
import cv2

from detection import scrfd, yolo5
from recognition import arcface, face_identifier
from reconstruction.deca import DECAReconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="3D face reconstruction pipeline from video", add_help=False)
    parser.add_argument('-p', '--path', type=str)
    return parser.parse_known_args()[0]


def pad_face(img, left, top, right, bottom):
    factor = 0.35
    pad_x = round((right - left) * factor / 2)
    pad_y = round((bottom - top) * factor / 2)

    left = max(0, left - pad_x)
    right = min(img.shape[1], right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(img.shape[0], bottom + pad_y)
    return left, top, right, bottom


def analyze_video(video_path):
    capture = cv2.VideoCapture(video_path)
    valid, frame = capture.read()

    if not valid:
        raise RuntimeError(video_path + " is not a valid video file")

    file_name = os.path.basename(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_time = max(1.0, 1000 / fps)

    detector = scrfd.SCRFaceDetector('model_files/scrfd_34g.onnx')
    # detector = yolo5.YOLOv5FaceDetector('model_files/yolov5l.pt')
    encoder = arcface.ArcFaceR100('model_files/arcface_r100.pth')
    identifier = face_identifier.FaceIdentifier(threshold=0.3)

    deca_file = 'model_files/deca_model.tar'
    flame_file = 'model_files/generic_model.pkl'
    albedo_file = 'model_files/FLAME_albedo_from_BFM.npz'
    deca = DECAReconstruction(deca_file, flame_file, albedo_file)

    key = cv2.waitKey(1)
    t1 = time.time_ns()

    out_directory = 'out'
    if os.path.exists(out_directory):
        shutil.rmtree(out_directory)
    os.makedirs(out_directory)

    while valid and key & 0xFF != ord('q'):
        bboxes = detector.detect(frame)

        for i, face in enumerate(bboxes):
            left, top, right, bottom = pad_face(frame, *face[:-1].astype(int))
            identity = -1

            if min(bottom-top, right-left) > 110:
                face_patch = frame[top:bottom+1, left:right+1]
                encoding = encoder.encode(face_patch)
                identity = identifier.get_identity(encoding)

                op_dict = deca.reconstruct(face_patch)
                face_nr = identifier.identities[identity].num_encodings
                obj_name = f'patch_{face_nr}'
                obj_dir = os.path.join(out_directory, f'id_{identity+1}', obj_name)
                os.makedirs(obj_dir, exist_ok=True)
                deca.save_obj(os.path.join(obj_dir, f'{obj_name}.obj'), op_dict)

            name = f'Person {identity + 1}'
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


if __name__ == '__main__':
    args = parse_args()
    analyze_video(args.path)
