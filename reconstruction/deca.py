import cv2
import numpy as np
import torch
from .DECA.decalib.deca import DECA
from .DECA.decalib.utils import config
'''
class DECAReconstruction:
    def __init__(self):
        pass
'''

class DECAReconstruction(DECA):
    def __init__(self, deca_file, flame_file, albedo_file=None):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = config.get_cfg_defaults()
        cfg.device = device_name
        cfg.pretrained_modelpath = deca_file
        cfg.model.flame_model_path = flame_file
        cfg.model.tex_path = albedo_file
        cfg.model.use_tex = albedo_file is not None

        super().__init__(cfg, torch.device(device_name))

    @torch.no_grad()
    def reconstruct(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255)

        codedict = super().encode(img)
        opdict, visdict = super().decode(codedict)
        return opdict