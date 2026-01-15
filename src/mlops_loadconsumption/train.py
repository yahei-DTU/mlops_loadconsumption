import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from model import Model
from data import MyDataset
import logging
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from visualize import plot_training_history

# Add configs folder to path
project_root = Path(__file__).resolve().parents[2]
configs_path = project_root / 'configs'
sys.path.insert(0, str(configs_path))

from config import (
    N_FEATURES, N_INPUT_TIMESTEPS, N_OUTPUT_TIMESTEPS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, API_KEY, COUNTRY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Training configuration - using global config values"""
    n_features = N_FEATURES
    n_timesteps = N_INPUT_TIMESTEPS
    n_outputs = N_OUTPUT_TIMESTEPS
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(__file__).parent.parent.parent / "data"
    sequences_path = data_path / "processed_data" / "sequences.pkl"
    model_path = Path(__file__).parent.parent.parent / "models" / "conv1d_model.pt"

def load_or_create_sequences(config: Config) -> dict:
    """Load sequences from cache or create them if they don't exist."""
    if config.sequences_path.exists():
        logger.info(f"Loading cached sequences from {config.sequences_path}")
        with open(config.sequences_path, 'rb') as f:
            sequences = pickle.load(f)
        logger.info("Sequences loaded successfully")
        return sequences
    else:
        logger.info("Cached sequences not found. Creating new sequences...")
        dataset = MyDataset(
            data_path=config.data_path,
            country=COUNTRY,
            api_key=API_KEY
        )
        dataset.preprocess()
        sequences = dataset.get_train_val_test_sequences()
        
        # Save sequences for future use
        config.sequences_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.sequences_path, 'wb') as f:
            pickle.dump(sequences, f)
        logger.info(f"Sequences saved to {config.sequences_path}")
        return sequences

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

def main():
    """Main training loop"""
    config = Config()
    
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Features: {config.n_features}")
    logger.info(f"  Input timesteps: {config.n_timesteps}")
    logger.info(f"  Output timesteps: {config.n_outputs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Device: {config.device}")
    logger.info("=" * 60)
    
    # Load or create sequences
    sequences = load_or_create_sequences(config)
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
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = Model(
        n_features=config.n_features,
        n_timesteps=config.n_timesteps,
        n_outputs=config.n_outputs
    ).to(config.device)
    
    logger.info(f"Model created on device: {config.device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss = validate(model, val_loader, criterion, config.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            config.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.model_path)
            logger.info(f"Best model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Plot training history
    plot_path = config.model_path.parent / 'training_history.png'
    plot_training_history(train_losses, val_losses, plot_path)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
