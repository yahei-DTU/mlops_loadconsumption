import torch
from torch import nn
import hydra
from omegaconf import DictConfig

class Model(nn.Module):
    """Convolutional Neural Network for Time Series Prediction."""
    def __init__(self, n_features: int, n_timesteps: int, n_outputs: int):
        """
        Initialize the CNN model.

        Args:
            n_features: Number of input features
            n_timesteps: Number of input timesteps
            n_outputs: Number of output timesteps to predict
        """
        super(Model, self).__init__()
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.n_outputs = n_outputs

        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        pooled_length = n_timesteps // 2
        self.dense1 = nn.Conv1d(64, 15, kernel_size=1)
        fc_input_size = 15 * pooled_length
        self.fc_final = nn.Linear(fc_input_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(x)
        x = self.dense1(x)
        x = x.view(x.size(0), -1)
        x = self.fc_final(x)
        return x

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Create and test model with config parameters."""
    model = Model(
        n_features=cfg.data.n_features,
        n_timesteps=cfg.data.n_input_timesteps,
        n_outputs=cfg.data.n_output_timesteps
    )
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, cfg.data.n_features, cfg.data.n_input_timesteps)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
