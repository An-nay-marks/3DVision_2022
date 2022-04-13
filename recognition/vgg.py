# code adapted from https://machinelearningmastery.com/how-to-perform-face-recognition-with-
# vggface2-convolutional-neural-network-in-keras/
import cv2
import numpy as np

from tqdm import tqdm
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from numpy import expand_dims


class VGGFaceClassifier:
    def __init__(self, threshold=0.3):
        self.classifier = VGGFace(model='resnet50')
        self.threshold = threshold
        self.sample_count = dict()

    def classify(self, img):
        img = img.astype('float32')
        img = cv2.resize(img, (224, 224))
        img = expand_dims(img, axis=0)
        sample = preprocess_input(img, version=2)
        class_code = decode_predictions(self.classifier.predict(sample))
        if class_code[0][0][1] < self.threshold:
            return -1

        # highest scored name and probability
        # class_code[0][0][1][0]
        identity = class_code[0][0][0][3:-1]
        if identity not in self.sample_count:
            self.sample_count[identity] = 0

        self.sample_count[identity] += 1
        return identity

    def classify_all(self, images):
        identities = []
        for img in tqdm(images):
            identities.append(self.classify(img))

        return np.asarray(identities), np.arange(len(identities))

    @staticmethod
    def get_name(identity):
        return "Unknown" if str(identity) == '-1' else identity

    def get_num_samples(self, identity):
        if identity not in self.sample_count:
            return 0

        return self.sample_count[identity]
