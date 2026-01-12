import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from mlops_loadconsumption.model import Model
from mlops_loadconsumption.data import MyDataset
import logging
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Config:
    """Training configuration"""
    n_features = 9
    n_timesteps = 4 * 24  # 4 days
    n_outputs = 24  # 1 day
    #batch_size = 32
    #epochs = 100
    #learning_rate = 0.0001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(__file__).parent.parent.parent / "data"
    model_path = Path(__file__).parent.parent.parent / "models" / "conv1d_model.pt"

def load_data(data_path):
    """Load and prepare data"""
    # TODO: Replace with your actual data loading logic
    # Example: data = pd.read_csv(data_path / "consumption.csv")
    logger.info(f"Loading data from {data_path}")
    pass

def create_sequences(data, n_timesteps, n_features):
    """
    Create sliding window sequences for supervised learning

    Args:
        data: numpy array of shape (n_samples, n_features)
        n_timesteps: length of input sequence
        n_features: number of features

    Returns:
        X, y: numpy arrays of shape (n_sequences, n_features, n_timesteps) and (n_sequences, n_outputs)
    """
    X, y = [], []
    for i in range(len(data) - n_timesteps):
        X.append(data[i:i + n_timesteps])
        y.append(data[i + n_timesteps:i + n_timesteps + Config.n_outputs, 0])  # Assuming first feature is consumption
    return np.array(X), np.array(y)

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

    return total_loss / len(val_loader)

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(cfg)

    lr = cfg.hyperparameters.lr
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs

    """Main training loop"""
    config = Config()

    # Create model
    model = Model(
        n_features=config.n_features,
        n_timesteps=config.n_timesteps,
        n_outputs=config.n_outputs
    ).to(config.device)

    logger.info(f"Model created on device: {config.device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Load data
    # X, y = load_data(config.data_path)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Create dummy data for testing
    X_train = np.random.randn(1000, config.n_timesteps, config.n_features).astype(np.float32)
    y_train = np.random.randn(1000, config.n_outputs).astype(np.float32)
    X_val = np.random.randn(200, config.n_timesteps, config.n_features).astype(np.float32)
    y_val = np.random.randn(200, config.n_outputs).astype(np.float32)

    # Convert to tensors and reshape for Conv1d (batch, features, timesteps)
    X_train = torch.from_numpy(X_train).transpose(1, 2)  # (1000, n_features, n_timesteps)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val).transpose(1, 2)
    y_val = torch.from_numpy(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    logger.info(f"Train set: {len(train_loader) * batch_size}, Val set: {len(val_loader) * batch_size}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss = validate(model, val_loader, criterion, config.device)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            config.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.model_path)
            logger.info(f"Best model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("Training completed!")

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    main()
