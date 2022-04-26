import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min

from dependencies.DECA.decalib.deca import DECA
from dependencies.DECA.decalib.utils.rotation_converter import batch_euler2axis
from dependencies.DECA.decalib.utils import config, util, tensor_cropper


class DECAFaceReconstruction(DECA):
    def __init__(self, deca_file, flame_file, albedo_file=None):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = config.get_cfg_defaults()
        cfg.device = device_name
        cfg.pretrained_modelpath = deca_file
        cfg.model.flame_model_path = flame_file
        cfg.model.tex_path = albedo_file
        cfg.model.use_tex = albedo_file is not None
        # cfg.rasterizer_type = "standard"

        super().__init__(cfg, torch.device(device_name))

    @torch.no_grad()
    def encode(self, img, average_shape=None):
        '''usual deca encoding, but can also use average shape parameters if given for more robust reconstruction'''
        img = self.transform_image_for_deca(img)
        if average_shape is None:
            codedict = super().encode(img)
        else:
            with torch.no_grad():
                flame_parameters = self.E_flame(img)
            detailcode = self.E_detail(img)
            codedict = self.decompose_code(flame_parameters, self.param_dict)
            codedict['images'] = img
            codedict['detail'] = detailcode
            codedict['shape'] = average_shape
            if self.cfg.model.jaw_type == 'euler':
                posecode = codedict['pose']
                euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
                posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
                codedict['pose'] = posecode
                codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict
    
    @torch.no_grad()
    def get_average_shape_param(self, imgs):
        '''returns the average shape parameter for all images'''
        flame_parameters = []
        for img in imgs:
            img = self.transform_image_for_deca(img)
            with torch.no_grad():
                parameters = self.E_flame(img)
            flame_parameters.append(parameters)
        
        # average flame and detail codes
        flame_parameters = torch.cat(flame_parameters, dim=0)
        mean_flame_parameters = torch.mean(flame_parameters, dim=0)
        mean_flame_parameters = mean_flame_parameters[None, :] # required dim = torch.Size([1, 236])
        codedict = self.decompose_code(mean_flame_parameters, self.param_dict)
        return codedict['shape']
    
    @torch.no_grad()
    def encode_average(self, imgs):
        """returns encoding average over all images
        code adapted from DECA
        """
        flame_parameters = []
        detail_parameters = []
        transformed_images = []
        for img in imgs:
            img = self.transform_image_for_deca(img)
            transformed_images.append(img)
            with torch.no_grad():
                parameters = self.E_flame(img)
            flame_parameters.append(parameters)
            detailcode = self.E_detail(img)
            detail_parameters.append(detailcode)
        
        # average flame and detail codes
        flame_parameters = torch.cat(flame_parameters, dim=0)
        mean_flame_parameters = torch.mean(flame_parameters, dim=0)
        mean_flame_parameters = mean_flame_parameters[None, :] # required dim = torch.Size([1, 236])
        detail_parameters = torch.cat(detail_parameters, dim=0)
        mean_detail_parameters = torch.mean(detail_parameters, dim=0)
        mean_detail_parameters = mean_detail_parameters[None, :] # required dim = torch.Size([1, 128])
        
        # get image closest to the average flame parameters for the decoder
        mean_flame_params_numpy = mean_flame_parameters.numpy()
        closest_idx, _ = pairwise_distances_argmin_min(mean_flame_params_numpy, flame_parameters.numpy())
        average_img = transformed_images[closest_idx[0]]
        
        # transform parameters into correct dictionary shape for decoder
        codedict = self.decompose_code(mean_flame_parameters, self.param_dict)
        codedict['images'] = average_img # image closest to cluster center for the decoder (albedo, texture, ...)
        codedict['detail'] = mean_detail_parameters
        if self.cfg.model.jaw_type == 'euler':
                posecode = codedict['pose']
                euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
                posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
                codedict['pose'] = posecode
                codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict
    
    @torch.no_grad()
    def transform_image_for_deca(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255)
        return img
    
    @torch.no_grad()
    def save_obj(self, filename, opdict):
        '''
        adapted from original DECA repository: added one detach
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        util.write_obj(filename, vertices, faces, 
                        texture=texture, 
                        uvcoords=uvcoords, 
                        uvfaces=uvfaces, 
                        normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:,:,[2,1,0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'), 
                        dense_vertices, 
                        dense_faces,
                        colors = dense_colors,
                        inverse_face_order=True)