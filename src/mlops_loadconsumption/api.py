"""FastAPI application for model inference."""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.mlops_loadconsumption.model import Model
from src.mlops_loadconsumption.monitor import check_drift_from_logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
predictions_log = []  # Store predictions for drift detection


def load_model():
    """Load the model from disk."""
    global model
    
    if model is not None:
        return  # Already loaded
    
    model_path = Path(__file__).parent.parent.parent / "models" / "conv1d_model.pt"
    
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = Model(n_features=12, n_timesteps=96, n_outputs=24)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    load_model()
    yield


app = FastAPI(
    title="Load Consumption Prediction API",
    description="API for electricity load forecasting",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]]  # Shape: (n_input_timesteps, n_features)


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[float]
    model_version: str = "1.0.0"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make a prediction using the trained model.
    
    Args:
        request: PredictionRequest with input features
        
    Returns:
        PredictionResponse with model predictions
    """
    # Load model if not already loaded (for testing)
    load_model()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensor
        X = torch.tensor(request.features, dtype=torch.float32)
        
        # Validate input shape
        if X.shape[0] != 96 or X.shape[1] != 12:
            raise ValueError(f"Expected shape (96, 12), got {X.shape}")
        
        # Add batch dimension and transpose for Conv1d
        X = X.unsqueeze(0)  # (1, timesteps, features)
        X = torch.transpose(X, 1, 2)  # (1, features, timesteps)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(X).squeeze(0).tolist()
        
        # Log the prediction
        predictions_log.append({
            "timestamp": datetime.now().isoformat(),
            "input_mean": float(np.mean(request.features)),
            "input_std": float(np.std(request.features)),
            "output_mean": float(np.mean(predictions)),
            "output_std": float(np.std(predictions))
        })
        
        logger.info("Prediction made successfully")
        
        return PredictionResponse(
            predictions=predictions,
            model_version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/logs")
async def get_logs():
    """Get prediction logs for monitoring."""
    return {"predictions": predictions_log}


@app.get("/check-drift")
async def check_drift():
    """Check for data drift in recent predictions."""
    if len(predictions_log) < 10:
        return {"message": "Not enough predictions to check drift (need at least 10)"}
    
    # Split logs: first half as reference, second half as current
    mid = len(predictions_log) // 2
    reference = predictions_log[:mid]
    current = predictions_log[mid:]
    
    results = check_drift_from_logs(reference, current)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)