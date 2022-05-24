# from dependencies.NOW_EVAL import check_predictions, compute_error
import os
import cv2
import numpy as np
import torch
import dependencies.DECA.decalib.utils.util as decautil

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from utils_3DV import read_lines_as_list, DEVICE, check_dataset
from .dataset import WeightingPatchesDataset, NoWDataset


class OptDataLoader(Dataset):
    def __init__(self, deca):
        self.reconstruction_score_file = "reconstruction_scores.txt"
        self.reconstruction_score_path = f"data/NoW_Dataset/final_release_version/{self.reconstruction_score_file}"
        self.deca = deca

        # check if NoW Dataset and all requirements available:
        check_dataset()
        img_paths_list = read_lines_as_list("data/NoW_Dataset/final_release_version/imagepathsvalidation.txt")
        self.data_set = WeightingPatchesDataset(img_paths=img_paths_list, gt_path=self.reconstruction_score_path)
    
    def getDataLoaders(self):
        # returns training and testing data loaders
        reconstruction_scores = self.get_reconstruction_scores()
        
        # return trainingDataloader, testingDataloader
        
    def get_reconstruction_scores(self):
        if os.path.exists(self.reconstruction_score_path):
            reconstruction_scores = np.loadtxt(self.reconstruction_score_path)
        else:
            # create reconstruction scores
        
            ''' NOW validation, code adapted from DECA Repo
            '''
            savefolder = "out/reconstructions_for_eval"
            os.makedirs(savefolder, exist_ok=True)
            self.deca.eval()
            # run now validation images
            dataset = NoWDataset(scale=1.6)
            dataloader = TorchDataLoader(dataset, batch_size=8, shuffle=False, num_workers=8,
                                         pin_memory=True, drop_last=False)
            faces = self.deca.flame.faces_tensor.cpu().numpy()
            for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
                images = batch['image'].to(DEVICE)
                imagename = batch['imagename']
                with torch.no_grad():
                    codedict = self.deca.encode(images)
                    _, visdict = self.deca.decode(codedict)
                    codedict['exp'][:] = 0.
                    codedict['pose'][:] = 0.
                    opdict, _ = self.deca.decode(codedict)
                #-- save results for evaluation
                verts = opdict['verts'].cpu().numpy()
                landmark_51 = opdict['landmarks3d_world'][:, 17:]
                landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
                landmark_7 = landmark_7.cpu().numpy()
                for k in range(images.shape[0]):
                    os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
                    # save mesh
                    decautil.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
                    # save 7 landmarks for alignment
                    np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
                    for vis_name in visdict.keys(): #['inputs', 'landmarks2d', 'shape_images']:
                        if vis_name not in visdict.keys():
                            continue
                        # import ipdb; ipdb.set_trace()
                        image = decautil.tensor2image(visdict[vis_name][k])
                        name = imagename[k].split('/')[-1]
                        # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
                        cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'), image)
                # visualize results to check
                decautil.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))
            
            # then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
            "python3 check_predictions.py <predicted_mesh_path> <predicted_mesh_landmark_path> <gt_scan_path> <gt_lmk_path> "
            # compute the error
            "python compute_error.py <dataset_folder> <predicted_mesh_folder> <validatton_or_test_set"
            return reconstruction_scores