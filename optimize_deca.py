import os
import torch
import argparse

from utils_3DV import ROOT_DIR
from reconstruction.trainer import Trainer
from reconstruction.model import OptimizerNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


def main(args):
    model = OptimizerNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, loss_function)

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train(num_epochs=args.epochs)
    checkpoint_path = os.path.join(ROOT_DIR, 'data', 'model_files', 'conv_predictor.pt')
    trainer.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main(parse_args())
