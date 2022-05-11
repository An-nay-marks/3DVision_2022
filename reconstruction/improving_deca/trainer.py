import random
import numpy as np

from torch.utils.data import DataLoader, Subset
from .dataloader import OptDataLoader
from utils_3DV import *
from reconstruct import initialize_deca


class Trainer:
    def __init__(self, model, optimizer, experiment_name, num_epochs, load_checkpoint_path=None, run_name=None,
                 split=0.8, batch_size=1, loss_function=None, scheduler=None, evaluation_interval=10,
                 num_samples_to_visualize=6, checkpoint_interval=50):
        """
        class for model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            attention_model: the attention_model to train
            optimizer: the optimizer to use
            experiment_name: name of the experiment to log this training run
            num_epochs: number of epochs, i.e. passes through the dataset, to train model for
            load_checkpoint_path (optional): if you want to continue a previous run, specify path of the last checkpoint
            run_name (optional): name of the run to log this training run under, default is current datetime
            split (optional): fraction of dataset provided by the DataLoader which to use for training rather than test (default is 0.8)
            batch_size (optional): number of samples to use per training iteration (None to use default)
            loss_function: loss function to use (None to use default) #TODO
            evaluation_interval: interval, in iterations, in which to perform an evaluation on the test set (default is 10)
            num_samples_to_visualize: number of samples to visualize predictions for during evaluation (default is 6)
            checkpoint_interval: interval, in iterations, in which to create model checkpoints (WARNING: None or 0 to discard model)
        """
        self.attention_model = model
        self.optimizer = optimizer
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.model = initialize_deca()

        self.dataloader = OptDataLoader(self.model)
        self.training_dataloader, self.test_dataloader = self.dataloader.getDataLoaders()

        self.run_name = run_name if run_name is not None else get_current_datetime_as_str()
        self.split = split
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.evaluation_interval = evaluation_interval
        self.num_samples_to_visualize = num_samples_to_visualize
        self.checkpoint_interval = checkpoint_interval
        self.do_checkpoint = self.checkpoint_interval is not None and self.checkpoint_interval > 0
        self.scheduler = scheduler
        self.load_checkpoint_path = load_checkpoint_path

    def train(self):
        """
        Trains the model
        """
        experiment_dir = os.path.join(CHECKPOINTS_DIR, self.experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, self.run_name)
        if self.do_checkpoint:
            for dir in [CHECKPOINTS_DIR, experiment_dir, checkpoint_dir]:
                if not os.path.exists(dir):
                    os.makedirs(dir)
        else:
            print('\nTraining started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print('Hyperparameters:')
            print(self._get_hyperparams())

            self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size)
            self.test_loader = self.dataloader.get_testing_dataloader(batch_size=1)

            print(f'Using device: {DEVICE}\n')
            self.attention_model = self.attention_model.to(DEVICE)
            self.model = self.model.to(DEVICE)

            if self.load_checkpoint_path is not None:
                self._load_checkpoint(self.load_checkpoint_path)

            callback_handler = Trainer.Callback(self, self.attention_model)

            for epoch in range(self.num_epochs):
                train_loss = self._train_step(callback_handler=callback_handler)
                test_loss = self._eval_step(self.test_loader)
                metrics = {'train_loss': train_loss, 'test_loss': test_loss}

                print('\nEpoch %i finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) % epoch)
                print('Metrics: %s\n' % str(metrics))
                # TODO: log metrics, iteration number, logfiles

            if self.do_checkpoint:
                # save final checkpoint
                self._save_checkpoint(self.attention_model, None, None, None)

            print('\nTraining finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
            # TODO: final logging

    def _train_step(self, callback_handler):
        self.attention_model.train()
        self.model.eval()
        opt = self.optimizer
        train_loss = 0
        for (x, y) in self.train_loader:
            x = x.to(DEVICE)
            weights = self.attention_model(x).tolist()
            dictionaries = []
            weighted_reconstructions = []
            for idx, patch in enumerate(x):
                encoding_dic = self.model.encode(patch)
                encoding_dic['shape'] *= weights[idx]  # TODO: test if this syntax even works with dictionaries
                reconstruction = self.model.decode(encoding_dic)
                dictionaries.append(encoding_dic.detach().cpu())
                weighted_reconstructions.append(reconstruction.detach().cpu())
                del encoding_dic
                del reconstruction
            del x
            y.to(DEVICE)
            weighted_reconstructions.to(DEVICE)
            loss = self.loss_function(weighted_reconstructions, y)
            with torch.no_grad():
                train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()
            del weighted_reconstructions
            del y
        train_loss /= len(self.train_loader.dataset)
        callback_handler.on_epoch_end()
        self.scheduler.step()
        return train_loss

    def _eval_step(self, test_loader):
        self.attention_model.eval()
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x = x.to(DEVICE)
                weights = self.attention_model(x).tolist()
                dictionaries = []
                weighted_reconstructions = []
                for idx, patch in enumerate(x):
                    encoding_dic = self.model.encode(patch)
                    encoding_dic['shape'] *= weights[idx]  # TODO: test if this syntax even works with dictionaries
                    reconstruction = self.model.decode(encoding_dic)
                    dictionaries.append(encoding_dic.detach().cpu())
                    weighted_reconstructions.append(reconstruction.detach().cpu())
                    del encoding_dic
                    del reconstruction
                del x
                y.to(DEVICE)
                weighted_reconstructions.to(DEVICE)
                loss = self.get_reconstruction_score(weighted_reconstructions, y)
                test_loss += loss.item()
                del weighted_reconstructions
                del y
            test_loss /= len(test_loader.dataset)
        return test_loss

    def _get_hyperparams(self):
        """
        Returns a dict of what is considered a hyperparameter
        """
        return {
            'split': self.split,
            'epochs': self.num_epochs,
            'batch size': self.batch_size,
            'optimizer': self.optimizer_or_lr,
            'loss function': self.loss_function,
        }

    # Visualizations are created using the "create_visualizations" functions of the Trainer
    def create_visualizations(self, file_path):
        # sample image indices to visualize
        # fix half of the samples, randomize other half
        # the first, fixed half of samples serves for comparison purposes across models/runs
        # the second, randomized half allows us to spot weaknesses of the model we might miss when
        # always visualizing the same samples
        num_to_visualize = self.num_samples_to_visualize
        num_fixed_samples = num_to_visualize // 2
        num_random_samples = num_to_visualize - num_fixed_samples
        # start sampling random indices from "num_fixed_samples + 1" to avoid duplicates
        # convert to np.array to allow subsequent slicing
        if num_to_visualize >= len(self.test_loader):
            # just visualize the entire test set
            indices = np.array(list(range(len(self.test_loader))))
        else:
            indices = np.array(list(range(num_fixed_samples)) + \
                               random.sample(range(num_fixed_samples + 1, len(self.test_loader)), num_random_samples))
        images = []
        # never exceed the given training batch size, else we might face memory problems
        vis_batch_size = min(num_to_visualize, self.batch_size)
        subset_ds = Subset(self.test_loader.dataset, indices)
        subset_dl = DataLoader(subset_ds, batch_size=vis_batch_size, shuffle=False)

        for (batch_xs, batch_ys) in subset_dl:
            batch_xs, batch_ys = batch_xs.to(DEVICE), batch_ys.numpy()
            output = self.attention_model(batch_xs)
            preds = (output >= self.segmentation_threshold).float().cpu().detach().numpy()
            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            self._fill_images_array(preds, batch_ys, images)

        self._save_image_array(images, file_path)

    def _save_checkpoint(self, model, epoch, epoch_iteration, total_iteration):
        if None not in [epoch, epoch_iteration, total_iteration]:
            checkpoint_path = f'{CHECKPOINT_DIR}/cp_ep-{"%05i" % epoch}_epit-{"%05i" % epoch_iteration}' + \
                              f'_step-{total_iteration}.pt'
        else:
            checkpoint_path = f'{CHECKPOINT_DIR}/cp_final.pt'
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': self.optimizer_or_lr.state_dict()
        }, checkpoint_path)

        # checkpoints should be logged right after their creation, in case training is
        # stopped/crashes *without* reaching the final checkpoint 
        # TODO: log checkpoints

    def _load_checkpoint(self, checkpoint_path):
        # this function may only be called after self.model has been moved
        # to DEVICE

        print(f'\n*** WARNING: resuming training from checkpoint "{checkpoint_path}" ***\n')
        final_checkpoint_path = checkpoint_path
        print(f'Loading checkpoint "{checkpoint_path}"...')  # log the supplied checkpoint_path here
        checkpoint = torch.load(final_checkpoint_path, map_location=DEVICE)
        self.attention_model.load_state_dict(checkpoint['model'])
        self.optimizer_or_lr.load_state_dict(checkpoint['optimizer'])
        print('Checkpoint loaded\n')
        os.remove(final_checkpoint_path)

    class Callback:
        """Callback as Helper class of Trainer
        gets initialized once every train() call
        gets called on each batch end and on each epoch end to save information at checkpoints
        """

        def __init__(self, trainer, model):
            super().__init__()
            self.model = model
            self.trainer = trainer
            self.run = trainer.run_name
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.epoch_idx = 0
            self.epoch_iteration_idx = 0
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_batch_end(self):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                precision, recall, f1_score = self.trainer.get_reconstruction_score()
                metrics = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
                print('Metrics at aggregate iteration %i (ep. %i, ep.-it. %i): %s'
                      % (self.iteration_idx, self.epoch_idx, self.epoch_iteration_idx, str(metrics)))
                # TODO: log metrics (loss, etc.) here

                if self.trainer.do_checkpoint \
                        and self.iteration_idx % self.trainer.checkpoint_interval == 0 \
                        and self.iteration_idx > 0:  # avoid creating checkpoints at iteration 0
                    self.trainer._save_checkpoint(self.trainer.model, self.epoch_idx, self.epoch_iteration_idx,
                                                  self.iteration_idx)

            self.iteration_idx += 1
            self.epoch_iteration_idx += 1

        def on_epoch_end(self):
            self.epoch_idx += 1
            self.epoch_iteration_idx = 0
