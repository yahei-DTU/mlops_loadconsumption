import torch
from torch import nn
import sys
from pathlib import Path
from config import N_FEATURES, N_INPUT_TIMESTEPS, N_OUTPUT_TIMESTEPS

# Add configs folder to path
project_root = Path(__file__).resolve().parents[2]
configs_path = project_root / "configs"
sys.path.insert(0, str(configs_path))


class Model(nn.Module):
    """Convolutional Neural Network for Time Series Prediction."""

    def __init__(
        self, n_features: int = N_FEATURES, n_timesteps: int = N_INPUT_TIMESTEPS, n_outputs: int = N_OUTPUT_TIMESTEPS
    ):
        super(Model, self).__init__()
        self.n_features = n_features
        self.n_timesteps = n_timesteps

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


if __name__ == "__main__":
    model = Model()
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, N_FEATURES, N_INPUT_TIMESTEPS)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
