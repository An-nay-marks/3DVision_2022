import os
import torch
import argparse

from utils_3DV import ROOT_DIR
from reconstruction.trainer import Trainer
from reconstruction.model import OptimizerCanny, OptimizerNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model', type=str, default= "standard", choices=["standard", "canny"])
    return parser.parse_args()


def main(args):
    if args.model == "standard":
        model = OptimizerNN()
        save_path = os.path.join(ROOT_DIR, 'data', 'model_files', 'conv_predictor.pt')
    elif args.model == "canny":
        model = OptimizerCanny()
        save_path = os.path.join(ROOT_DIR, 'data', 'model_files', 'canny_predictor.pt')
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
    loss_function = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, loss_function)

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train(num_epochs=args.epochs)
    trainer.save_checkpoint(save_path)
    print(f'Final checkpoint saved at {save_path}')


if __name__ == "__main__":
    main(parse_args())
