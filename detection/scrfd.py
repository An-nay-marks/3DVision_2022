import onnxruntime
from insightface.detection.scrfd.tools.scrfd import SCRFD


class SCRFaceDetector(SCRFD):
    def __init__(self, model_file):
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'])
        session = onnxruntime.InferenceSession(model_file, None, providers)
        super().__init__(model_file, session)
        self.prepare(0, input_size=(640, 640))

    def detect(self, frame, thresh=0.6, input_size=None, max_num=0, metric='default'):
        rgb_frame = frame[:, :, ::-1]
        return super().detect(rgb_frame, thresh, input_size, max_num, metric)[0]
