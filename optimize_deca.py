from reconstruction.improving_deca.trainer import Trainer
from reconstruction.improving_deca.model import OptimizerNN
from torch.optim import Adam

if __name__ == "__main__":
    model = OptimizerNN(2)
    optimizer = Adam(model.parameters())
    trainer = Trainer(model=model, optimizer=optimizer, experiment_name="test", num_epochs=2, checkpoint_interval=1)
