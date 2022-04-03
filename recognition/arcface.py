import cv2
import numpy as np
import torch

from detection.insightface.recognition.arcface_torch.backbones import get_model


class ArcFace:
    def __init__(self, model_file, name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(name, fp16=False).to(self.device)
        self.model.load_state_dict(torch.load(model_file, self.device))
        self.model.eval()
        self.model(torch.rand(1, 3, 112, 112, device=self.device).sub_(0.5).div_(0.5))  # warm up

    @torch.no_grad()
    def encode(self, img):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255).sub_(0.5).div_(0.5)
        return self.model(img).cpu().numpy()[0]


class ArcFaceR18(ArcFace):
    def __init__(self, model_file):
        super().__init__(model_file, "r18")


class ArcFaceR34(ArcFace):
    def __init__(self, model_file):
        super().__init__(model_file, "r34")


class ArcFaceR50(ArcFace):
    def __init__(self, model_file):
        super().__init__(model_file, "r50")


class ArcFaceR100(ArcFace):
    def __init__(self, model_file):
        super().__init__(model_file, "r100")
