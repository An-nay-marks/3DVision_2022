# code adapted from https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
from keras_vggface.vggface import VGGFace
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from numpy import expand_dims

class VGGEncoderAndClassifier():
    def __init__(self, threshold=0.3):
        self.classifier = VGGFace(model ='resnet50')
        self.threshold = threshold
    
    def classify(self, img):
        img = img.astype('float32')
        img = cv2.resize(img, (224,224))
        img = expand_dims(img, axis=0)
        sample = preprocess_input(img, version=2)
        class_code = decode_predictions(self.classifier.predict(sample))
        if class_code[0][0][1] < self.threshold:
            print(class_code[0])
            return -1,-1
        # highest scored name and probability
        return class_code[0][0][0].split(' ')[1], class_code[0][0][1]