import sys
import cv2

from recognition import arcface
from recognition.face_identifier import FaceIdentifier


def main(path1, path2):
    encoder = arcface.ArcFaceR50('model_files/arcface_r50.pth')

    img1 = cv2.imread(path1)
    enc1 = encoder.encode(img1)

    img2 = cv2.imread(path2)
    enc2 = encoder.encode(img2)

    sim = FaceIdentifier.face_similarity(enc1, enc2)
    print(sim)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
