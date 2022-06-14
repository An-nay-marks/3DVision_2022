import cv2
import numpy as np
import torch

from dependencies.DECA.decalib.deca import DECA
from dependencies.DECA.decalib.utils import config
from reconstruction.model import OptimizerNN
from utils_3DV import DEVICE


class DECAFaceReconstruction(DECA):
    def __init__(self, deca_file, flame_file, albedo_file, merge_fn, optimizer_file=None):
        cfg = config.get_cfg_defaults()
        cfg.device = DEVICE
        cfg.pretrained_modelpath = deca_file
        cfg.model.flame_model_path = flame_file
        cfg.model.tex_path = albedo_file
        cfg.model.use_tex = albedo_file is not None
        super().__init__(cfg, DEVICE)
        self.merge_fn = merge_fn

        if merge_fn == 'predictive':
            self.model = OptimizerNN(optimizer_file).to(DEVICE)

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255)
        return img

    @torch.no_grad()
    def reconstruct(self, img, preprocessing=True):
        if preprocessing:
            img = self.preprocess(img)
        code_dict = self.encode(img)
        op_dict, _ = self.decode(code_dict)
        return op_dict

    @torch.no_grad()
    def reconstruct_multiple(self, images, preprocessing=True):
        reconstructions = []

        if preprocessing:
            images = [self.preprocess(img) for img in images]

        if self.merge_fn == 'single':
            for img in images:
                reconstruction = self.reconstruct(img, False)
                reconstructions.append(reconstruction)
        else:
            encodings = []
            for img in images:
                enc = self.encode(img)
                encodings.append(enc)

            if self.merge_fn == "mean":
                code_dict = self._combine_all_params(encodings)
                reconstruction, _ = self.decode(code_dict)
                reconstructions.append(reconstruction)
            elif self.merge_fn == "mean_shape":
                for code_dict in self._average_shape_params(encodings):
                    reconstruction, _ = self.decode(code_dict)
                    reconstructions.append(reconstruction)
            else:  # predictive
                images = torch.cat(images).to(DEVICE)
                scores = self.model(images).squeeze(1)
                weights = torch.softmax(-scores / 1000, dim=0)
                code_dict = self._combine_all_params(encodings, weights)
                reconstruction, _ = self.decode(code_dict)
                reconstructions.append(reconstruction)

        return reconstructions

    @staticmethod
    def _get_weighted_param_avg(encodings, key, weights=None):
        if weights is None:
            weights = torch.ones(len(encodings), device=DEVICE) / len(encodings)

        all_params = torch.cat([enc[key] for enc in encodings], dim=0)
        return torch.sum(torch.mul(weights, all_params.T).T, dim=0).unsqueeze(0)

    @staticmethod
    def _get_representative_sample(encodings, key):
        all_params = torch.cat([enc[key] for enc in encodings], dim=0)
        nearest_idx = (all_params - torch.mean(all_params, dim=0)).abs().sum(dim=1).argmin()
        return int(nearest_idx)

    @staticmethod
    def _combine_all_params(encodings, weights=None):
        code_dict = dict()

        exceptions = ['tex', 'images']
        param_keys = encodings[0].keys()
        for key in [k for k in param_keys if k not in exceptions]:
            code_dict[key] = DECAFaceReconstruction._get_weighted_param_avg(encodings, key, weights)

        # use image of most representative sample
        nearest_idx = DECAFaceReconstruction._get_representative_sample(encodings, 'shape')
        for key in exceptions:
            code_dict[key] = encodings[nearest_idx][key]
        return code_dict

    @staticmethod
    def _average_shape_params(encodings):
        new_encodings = []
        for enc in encodings:
            code_dict = enc.copy()
            code_dict['shape'] = DECAFaceReconstruction._get_weighted_param_avg(encodings, 'shape')
            new_encodings.append(code_dict)
        return new_encodings
