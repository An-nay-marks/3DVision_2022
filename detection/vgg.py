from keras_vggface.vggface import VGGFace

class VGGEncoder():
    def __init__(self):
        self.encoder_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')