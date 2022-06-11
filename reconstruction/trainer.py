import contextlib
import tempfile

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from reconstruction.now import NoWDataset
from utils_3DV import *


class Trainer:
    def __init__(self, model, optimizer, loss_function, checkpoint_path=None, name=None, split=0.8, batch_size=8,
                 evaluation_interval=2, checkpoint_interval=5):
        """
        class for model trainers.
        Args:
            model: the model to train
            optimizer: the optimizer to use
            loss_function: loss function to use
            checkpoint_path (optional): if you want to continue a previous run, specify path of the last checkpoint
            name (optional): name of the run to log this training run under, default is current datetime
            split (optional): fraction of dataset to use for training rather than test
            batch_size (optional): number of samples to use per training iteration
            evaluation_interval: how often to run evaluation (in epochs)
            checkpoint_interval: how often to save model checkpoints (in epochs)
        """
        self.model = model.to(DEVICE)
        self.optimizer = optimizer

        self.run_name = name or get_current_datetime_as_str()

        self.epoch = 0
        self.train_ratio = split
        self.batch_size = batch_size

        self.loss_function = loss_function
        self.evaluation_interval = evaluation_interval
        self.checkpoint_interval = checkpoint_interval

        dist_path = os.path.join(ROOT_DIR, 'data', 'now_dist.npy')

        if os.path.exists(dist_path):
            self.deca = None
        else:
            from reconstruct import initialize_deca
            self.deca = initialize_deca('single')
            dist_path = None

        train_data = NoWDataset('train', self.train_ratio, dist_path=dist_path)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        test_data = NoWDataset('test', self.train_ratio, dist_path=dist_path)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _get_hyperparams(self):
        """
        Returns dict of all hyperparameters for convenience
        """
        return {
            'split': self.train_ratio,
            'batch size': self.batch_size,
            'optimizer': self.optimizer,
            'loss function': self.loss_function,
        }

    def train(self, num_epochs):
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, self.run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        print('')
        print('Training started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
        print('Hyperparameters:')
        print(self._get_hyperparams())

        self.model.train()

        for self.epoch in range(num_epochs):
            print(f'Epoch {self.epoch+1}/{num_epochs}')
            with contextlib.redirect_stderr(tempfile.TemporaryFile(mode='w+')):
                train_loss = self._train_step()
                print(f'Training loss: {train_loss}')

                if self.epoch % self.evaluation_interval == 0:
                    test_loss = self._eval_step()
                    print(f'Evaluation loss: {test_loss}')

                if self.epoch % self.checkpoint_interval == 0:
                    path = os.path.join(checkpoint_dir, f'cp_ep-{self.epoch:05}.pt')
                    self._save_checkpoint(path)
                    print(f'Checkpoint saved at {path}')

                print('')

        print('Training finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))

    def _process_batch(self, batch):
        images = batch[0]
        scores = self.model(images).squeeze(1)

        if len(batch) == 3:
            gt_mesh_paths, gt_lmk_paths = batch[1:]

            encoding = self.deca.encode(images)
            reconstruction, _ = self.deca.decode(encoding)
            faces = self.deca.flame.faces_tensor.cpu().numpy()

            verts = reconstruction['verts'].cpu().numpy()
            landmark_51 = reconstruction['landmarks3d_world'][:, 17:]
            landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
            landmark_7 = landmark_7.cpu().numpy()
            distances = torch.zeros(images.shape[0])

            sys.path.append('dependencies/now_evaluation')
            from psbody.mesh import Mesh
            from dependencies.now_evaluation.compute_error import compute_error_metric

            for k in range(images.shape[0]):
                pred_mesh = Mesh(basename=f'img_{k}', v=verts[k], f=faces)
                pred_lmk = landmark_7[k]
                landmark_dist = compute_error_metric(gt_mesh_paths[k], gt_lmk_paths[k], pred_mesh, pred_lmk)
                distances[k] = np.sum(landmark_dist)
        else:
            distances = torch.Tensor(batch[1])

        distances = distances.to(DEVICE)
        return self.loss_function(scores, distances)

    def _train_step(self):
        train_loss = 0
        self.model.train()

        for batch in tqdm(self.train_loader, file=sys.stdout):
            loss = self._process_batch(batch)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return train_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval_step(self):
        test_loss = 0
        self.model.eval()

        for batch in tqdm(self.test_loader, file=sys.stdout):
            loss = self._process_batch(batch)
            test_loss += loss.item()

        return test_loss / len(self.test_loader)

    def _save_checkpoint(self, checkpoint_path):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        print(f'Loading checkpoint "{checkpoint_path}"...')
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Checkpoint loaded, resuming from epoch {self.epoch}.')
