"""Visualization utilities for model training and evaluation."""
import plotly.graph_objects as go
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_training_history(train_losses: list, val_losses: list, save_path: Path) -> None:
    """
    Plot training and validation losses over epochs using Plotly.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot (HTML format)
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    fig = go.Figure()
    
    # Add training loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Add validation loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Val Loss',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600,
        font=dict(size=12),
        legend=dict(
            x=0.7,
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    logger.info(f"Training history plot saved to {save_path}")
    
    # Also display in browser
    fig.show()