import argparse
import contextlib
import os.path
import sys
import tempfile

import numpy as np
from psbody.mesh import Mesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_3DV import MERGE_STRATEGIES, ROOT_DIR
from reconstruct import initialize_deca
from reconstruction.now import SubjectBasedNoWDataSet

sys.path.append('dependencies/now_evaluation')
from dependencies.now_evaluation.compute_error import compute_error_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', choices=MERGE_STRATEGIES, default=MERGE_STRATEGIES[0],
                        help=f'Method with which to combine face reconstructions.')
    return parser.parse_known_args()[0]


def main(args):
    data_set = SubjectBasedNoWDataSet('imagepathsvalidation.txt')
    data_loader = DataLoader(data_set, shuffle=False)
    deca = initialize_deca(args.merge)

    with contextlib.redirect_stderr(tempfile.TemporaryFile('w+')):
        all_means = []

        for subject in tqdm(data_loader, file=sys.stdout):
            images, gt_mesh_path, gt_lmk_path = subject
            reconstructions = deca.reconstruct_multiple(images, False)
            faces = deca.flame.faces_tensor.cpu().numpy()

            for reconstruction in reconstructions:
                verts = reconstruction['verts'].cpu().numpy()
                landmark_51 = reconstruction['landmarks3d_world'][:, 17:]
                landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
                landmark_7 = landmark_7.cpu().numpy()

                pred_mesh = Mesh(basename='img', v=verts[0], f=faces)
                landmark_dist = compute_error_metric(gt_mesh_path[0], gt_lmk_path[0], pred_mesh, landmark_7[0])
                all_means.append(np.mean(landmark_dist))

        # np.save(os.path.join(ROOT_DIR, 'data', 'now_dist.npy'), all_means)
        print(f'median: {np.median(all_means)}')
        print(f'mean:   {np.mean(all_means)}')
        print(f'std:    {np.std(all_means)}')


if __name__ == '__main__':
    main(parse_args())
