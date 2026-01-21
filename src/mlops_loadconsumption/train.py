import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from mlops_loadconsumption.model import Model
from mlops_loadconsumption.data import MyDataset
import logging
import pickle
from sklearn.preprocessing import StandardScaler
from mlops_loadconsumption.visualize import plot_training_history
import hydra
from omegaconf import DictConfig
import wandb
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT", "mlops-loadconsumption")
WANDB_ENTITY_NAME = os.getenv("WANDB_ENTITY", None)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_or_create_sequences(cfg: DictConfig, data_path: Path, sequences_path: Path) -> dict:
    """Load sequences from cache or create them if they don't exist."""
    if sequences_path.exists():
        logger.info(f"Loading cached sequences from {sequences_path}")
        with open(sequences_path, 'rb') as f:
            sequences = pickle.load(f)
        logger.info("Sequences loaded successfully")
        return sequences
    else:
        logger.info("Cached sequences not found. Creating new sequences...")
        dataset = MyDataset(
            n_input_timesteps=cfg.data.n_input_timesteps,
            n_output_timesteps=cfg.data.n_output_timesteps,
            train_size=cfg.split.train_size,
            val_size=cfg.split.val_size,
            test_size=cfg.split.test_size,
            data_path=data_path,
            country=cfg.api.country,
            api_key=cfg.api.key
        )
        dataset.preprocess()
        sequences = dataset.get_train_val_test_sequences()

        # Save sequences for future use
        sequences_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sequences_path, 'wb') as f:
            pickle.dump(sequences, f)
        logger.info(f"Sequences saved to {sequences_path}")
        return sequences

def compute_regression_accuracy(predictions: torch.Tensor, targets: torch.Tensor, tolerance: float = 0.1) -> float:
    """Fraction of predictions within a relative tolerance of targets."""
    rel_error = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
    within_tol = (rel_error <= tolerance).float()
    return within_tol.mean().item()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    start_step: int,
) -> tuple[float, int]:
    """Train for one epoch and log per-step metrics to wandb."""
    model.train()
    total_loss = 0.0
    global_step = start_step

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        acc = compute_regression_accuracy(outputs.detach(), y.detach())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        # Per-step logging
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/accuracy": acc,
                "train/step": global_step,
            },
            step=global_step,
        )

        if batch_idx % 10 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc:.4f}"
            )

    return total_loss / len(train_loader), global_step

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
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

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training loop"""
    # Initialize wandb
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME, config=dict(cfg), mode="online")

    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data"
    sequences_path = data_path / "processed_data" / "sequences.pkl"
    model_path = project_root / "models" / "conv1d_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Features: {cfg.data.n_features}")
    logger.info(f"  Input timesteps: {cfg.data.n_input_timesteps}")
    logger.info(f"  Output timesteps: {cfg.data.n_output_timesteps}")
    logger.info(f"  Batch size: {cfg.training.batch_size}")
    logger.info(f"  Learning rate: {cfg.training.learning_rate}")
    logger.info(f"  Epochs: {cfg.training.epochs}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)

    # Load or create sequences
    sequences = load_or_create_sequences(cfg, data_path, sequences_path)
    X_train = sequences['X_train']
    y_train = sequences['y_train']
    X_val = sequences['X_val']
    y_val = sequences['y_val']

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Convert TensorFlow tensors to NumPy
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_val_np = X_val.numpy()
    y_val_np = y_val.numpy()

    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Reshape for scaling: (batch, timesteps, features) -> (batch*timesteps, features)
    X_train_reshaped = X_train_np.reshape(-1, X_train_np.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(X_train_np.shape)

    X_val_reshaped = X_val_np.reshape(-1, X_val_np.shape[-1])
    X_val_scaled = scaler_X.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled.reshape(X_val_np.shape)

    # Scale targets
    y_train_reshaped = y_train_np.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped)
    y_train_scaled = y_train_scaled.reshape(y_train_np.shape)

    y_val_reshaped = y_val_np.reshape(-1, 1)
    y_val_scaled = scaler_y.transform(y_val_reshaped)
    y_val_scaled = y_val_scaled.reshape(y_val_np.shape)

    logger.info(f"X_train scaled - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    logger.info(f"y_train scaled - mean: {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}")

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train_scaled).float()
    y_train = torch.from_numpy(y_train_scaled).float()
    X_val = torch.from_numpy(X_val_scaled).float()
    y_val = torch.from_numpy(y_val_scaled).float()

    # Transpose for Conv1d: (batch, features, timesteps)
    X_train = torch.transpose(X_train, 1, 2)
    X_val = torch.transpose(X_val, 1, 2)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = Model(
        n_features=cfg.data.n_features,
        n_timesteps=cfg.data.n_input_timesteps,
        n_outputs=cfg.data.n_output_timesteps
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created on device: {device}")
    logger.info(f"Total parameters: {param_count}")

    # Log key hyperparameters/model info
    wandb.config.update(
        {
            "model/parameter_count": param_count,
            "model/n_features": cfg.data.n_features,
            "model/n_timesteps": cfg.data.n_input_timesteps,
            "model/n_outputs": cfg.data.n_output_timesteps,
            "training/batch_size": cfg.training.batch_size,
            "training/learning_rate": cfg.training.learning_rate,
            "training/epochs": cfg.training.epochs,
        },
        allow_val_change=True,
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    global_step = 0

    for epoch in range(cfg.training.epochs):
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, global_step
        )
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Best model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Plot training history
    plot_path = model_path.parent / 'training_history.png'
    plot_training_history(train_losses, val_losses, plot_path)

    logger.info("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
