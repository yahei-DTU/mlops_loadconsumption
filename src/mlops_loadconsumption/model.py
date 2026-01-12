import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_features, n_timesteps, n_outputs=48):
        super(Model, self).__init__()
        self.n_features = n_features
        self.n_timesteps = n_timesteps

        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        pooled_length = n_timesteps // 2 # After pooling: (batch, 64, n_timesteps//2)
        self.dense1 = nn.Conv1d(64, 15, kernel_size=1) # Dense(15) applied to each timestep: Conv1d with kernel_size=1
        fc_input_size = 15 * pooled_length # Flatten: (batch, 15 * pooled_length)
        self.fc_final = nn.Linear(fc_input_size, n_outputs)

    def forward(self, x):
        # x shape: (batch, n_features, n_timesteps)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(x)  # (batch, 64, n_timesteps//2)
        x = self.dense1(x)  # (batch, 15, n_timesteps//2)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 15 * n_timesteps//2)
        x = self.fc_final(x)  # (batch, n_outputs)
        return x

if __name__ == "__main__":
    n_features = 9
    n_timesteps = 4*24  # 4 days
    n_outputs = 24  # 1 day

    model = Model(n_features, n_timesteps, n_outputs)
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, n_features, n_timesteps)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
