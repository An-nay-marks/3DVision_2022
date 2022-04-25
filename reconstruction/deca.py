import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
    def encode(self, img):
        img = self.transform_image_for_deca(img)
        code_dict = super().encode(img)
        return code_dict
    
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
        imgs = torch.cat(transformed_images)
        
        # average flame and detail codes
        flame_parameters = torch.cat(flame_parameters, dim=0)
        mean_flame_parameters = torch.mean(flame_parameters, dim=0)
        mean_flame_parameters = mean_flame_parameters[None, :] # required dim = torch.Size([1, 236])
        detail_parameters = torch.cat(detail_parameters, dim=0)
        mean_detail_parameters = torch.mean(detail_parameters, dim=0)
        mean_detail_parameters = mean_detail_parameters[None, :] # required dim = torch.Size([1, 128])
        
        # transform parameters into correct dictionary shape for decoder
        codedict = self.decompose_code(mean_flame_parameters, self.param_dict)
        codedict['images'] = imgs
        codedict['detail'] = mean_detail_parameters
        if self.cfg.model.jaw_type == 'euler':
                posecode = codedict['pose']
                euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
                posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
                codedict['pose'] = posecode
                codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict
    
    @torch.no_grad()
    def decode_average(self, codedict, rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True,
                render_orig=False, original_image=None, tform=None):
        '''adapted from original DECA, can now handle multiple images and averages their albedo shape'''
        images = codedict['images']
        batch_size = images[0].shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedos = []
            for img in images:
                albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=img.device) 
            albedos_torch = torch.cat(albedos, dim=0)
            albedo = torch.mean(albedos_torch, dim=0) # mean albedo
        landmarks3d_world = landmarks3d.clone()
        
        # average images
        images = torch.mean(images, axis=0)
        images = images[None] # add dimensionality

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }

        ## rendering
        if rendering:
            ops = self.render(verts, trans_verts, albedo, codedict['light'])
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
        
        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo
            
        if use_detail:
            uv_z = self.D_detail(torch.cat([codedict['pose'][:,3:], codedict['exp'], codedict['detail']], dim=1))
            if iddict is not None:
                uv_z = self.D_detail(torch.cat([iddict['pose'][:,3:], iddict['exp'], codedict['detail']], dim=1))
            uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
            uv_texture = albedo*uv_shading

            opdict['uv_texture'] = uv_texture 
            opdict['normals'] = ops['normals']
            opdict['uv_detail_normals'] = uv_detail_normals
            opdict['displacement_map'] = uv_z+self.fixed_uv_dis[None,None,:,:]
        
        if vis_lmk:
            landmarks3d_vis = self.visofp(ops['transformed_normals'])#/self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if return_vis:
            h, w = self.image_size, self.image_size
            background = None
            ## render shape
            shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w, images=background, return_grid=True)
            detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)*alpha_images
            shape_detail_images = self.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images, h=h, w=w, images=background)
            
            ## extract texture
            ## TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_pverts = self.render.world2uv(trans_verts)
            uv_gt = F.grid_sample(images, uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear')
            if self.cfg.model.use_tex:
                ## TODO: poisson blending should give better-looking results
                uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask))
            else:
                uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (torch.ones_like(uv_gt[:,:3,:,:])*(1-self.uv_face_eye_mask)*0.7)
            
            opdict['uv_texture_gt'] = uv_texture_gt
            visdict = {
                'inputs': images, 
                'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                'shape_images': shape_images,
                'shape_detail_images': shape_detail_images
            }
            if self.cfg.model.use_tex:
                visdict['rendered_images'] = ops['images']

            return opdict, visdict

        else:
            return opdict
    
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