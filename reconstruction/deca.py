import cv2
import numpy as np
import torch

from dependencies.DECA.decalib.deca import DECA
from dependencies.DECA.decalib.utils import config, util
from utils_3DV import DEVICE


class DECAFaceReconstruction(DECA):
    def __init__(self, deca_file, flame_file, albedo_file, merge_fn):
        cfg = config.get_cfg_defaults()
        cfg.device = DEVICE
        cfg.pretrained_modelpath = deca_file
        cfg.model.flame_model_path = flame_file
        cfg.model.tex_path = albedo_file
        cfg.model.use_tex = albedo_file is not None
        super().__init__(cfg, DEVICE)
        self.merge_fn = merge_fn

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255)
        return img

    @torch.no_grad()
    def reconstruct(self, img):
        img = self.preprocess(img)
        code_dict = self.encode(img)
        op_dict, _ = self.decode(code_dict)
        return op_dict

    @torch.no_grad()
    def reconstruct_multiple(self, images):
        reconstructions = []

        if self.merge_fn == 'single':
            for img in images:
                reconstruction = self.reconstruct(img)
                reconstructions.append(reconstruction)
        else:
            encodings = []
            for img in images:
                enc = self.encode(self.preprocess(img))
                encodings.append(enc)

            if self.merge_fn == "mean":
                code_dict = self._average_all_parameters(encodings)
                reconstruction, _ = self.decode(code_dict)
                reconstructions.append(reconstruction)
            else:  # mean_shape
                for code_dict in self._average_shape_params(encodings):
                    reconstruction, _ = self.decode(code_dict)
                    reconstructions.append(reconstruction)

        return reconstructions

    @staticmethod
    def _get_parameter_mean(encodings, key):
        all_params = torch.cat([enc[key] for enc in encodings], dim=0)
        return torch.mean(all_params, dim=0)[None, :]

    @staticmethod
    def _get_representative_sample(encodings, key):
        all_params = torch.cat([enc[key] for enc in encodings], dim=0)
        nearest_idx = (all_params - torch.mean(all_params, dim=0)).abs().sum(dim=1).argmin()
        return int(nearest_idx)

    @staticmethod
    def _average_all_parameters(encodings):
        code_dict = dict()
        param_keys = encodings[0].keys()
        for key in [k for k in param_keys if k != 'images']:
            code_dict[key] = DECAFaceReconstruction._get_parameter_mean(encodings, key)

        # use image of most representative sample
        nearest_idx = DECAFaceReconstruction._get_representative_sample(encodings, 'shape')
        code_dict['images'] = encodings[nearest_idx]['images']
        return code_dict

    @staticmethod
    def _average_shape_params(encodings):
        new_encodings = []
        for enc in encodings:
            code_dict = enc.copy()
            code_dict['shape'] = DECAFaceReconstruction._get_parameter_mean(encodings, 'shape')
            new_encodings.append(code_dict)
        return new_encodings

    def save_obj(self, filename, opdict):
        """
        adapted from original DECA repository: added one detach
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        """
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
        util.write_obj(filename, vertices, faces,
                       texture=texture,
                       uvcoords=uvcoords,
                       uvfaces=uvfaces,
                       normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:, :, [2, 1, 0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map,
                                                                       texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'), dense_vertices, dense_faces,
                       colors=dense_colors, inverse_face_order=True)
