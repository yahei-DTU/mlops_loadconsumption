import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from mlops_loadconsumption.model import Model


class TestModelTraining:
    """Integration tests for model training."""

    @pytest.fixture
    def model_config(self) -> dict:
        """Provide model configuration."""
        return {
            'n_features': 5,
            'n_timesteps': 168,
            'n_outputs': 24
        }

    @pytest.fixture
    def model(self, model_config: dict) -> Model:
        """Create a model instance."""
        return Model(**model_config)

    @pytest.fixture
    def sample_data(self, model_config: dict) -> DataLoader:
        """Create sample training data."""
        batch_size = 32
        X = torch.randn(batch_size, model_config['n_features'], model_config['n_timesteps'])
        y = torch.randn(batch_size, model_config['n_outputs'])
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=16)

    def test_model_training_reduces_loss(self, model: Model, sample_data: DataLoader) -> None:
        """Test that model training reduces loss over iterations."""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        initial_loss = None
        final_loss = None

        model.train()
        for epoch in range(5):
            epoch_loss = 0.0
            for X_batch, y_batch in sample_data:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch == 0:
                initial_loss = epoch_loss
            if epoch == 4:
                final_loss = epoch_loss

        assert final_loss < initial_loss, "Loss should decrease during training"

    def test_model_can_overfit_small_batch(self, model_config: dict) -> None:
        """Test that model can overfit a single batch (sanity check)."""
        model = Model(**model_config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        X = torch.randn(4, model_config['n_features'], model_config['n_timesteps'])
        y = torch.randn(4, model_config['n_outputs'])

        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < 0.1, f"Model should overfit small batch, but loss is {final_loss}"

    def test_model_gradients_flow(self, model: Model, sample_data: DataLoader) -> None:
        """Test that gradients flow through the model."""
        criterion = nn.MSELoss()

        X_batch, y_batch = next(iter(sample_data))
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
