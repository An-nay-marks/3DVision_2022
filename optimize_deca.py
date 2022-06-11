import torch

from reconstruction.trainer import Trainer
from reconstruction.model import OptimizerNN


def main():
    model = OptimizerNN()
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, loss_function)
    trainer.train(num_epochs=50)


if __name__ == "__main__":
    main()
