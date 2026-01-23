"""Tests for the FastAPI application."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_loadconsumption.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model_loaded" in response.json()


def test_predict_with_valid_input(client):
    """Test prediction endpoint with valid input."""
    # Generate valid input: 96 timesteps, 12 features
    features = np.random.randn(96, 12).tolist()
    print(features)
    
    response = client.post(
        "/predict",
        json={"features": features}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "model_version" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 24  # Model outputs 24 timesteps


def test_predict_with_invalid_shape(client):
    """Test prediction endpoint with invalid input shape."""
    # Generate invalid input: wrong number of timesteps
    features = np.random.randn(50, 12).tolist()
    
    response = client.post(
        "/predict",
        json={"features": features}
    )
    
    assert response.status_code == 400
    assert "Expected shape" in response.json()["detail"]


def test_predict_with_wrong_features(client):
    """Test prediction endpoint with wrong number of features."""
    # Generate invalid input: wrong number of features
    features = np.random.randn(96, 10).tolist()
    
    response = client.post(
        "/predict",
        json={"features": features}
    )
    
    assert response.status_code == 400


def test_predict_response_format(client):
    """Test that prediction response has correct format."""
    features = np.random.randn(96, 12).tolist()
    
    response = client.post(
        "/predict",
        json={"features": features}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, dict)
    assert "predictions" in data
    assert "model_version" in data
    assert data["model_version"] == "1.0.0"
    
    # Check predictions are floats
    assert all(isinstance(p, (int, float)) for p in data["predictions"])