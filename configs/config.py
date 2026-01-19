"""Global configuration constants for the MLOps load consumption project."""

# Data configuration
N_FEATURES = 12
N_INPUT_TIMESTEPS = 96  # 4 days
N_OUTPUT_TIMESTEPS = 24

# Model configuration
MODEL_NAME = "load_consumption_model"
MODEL_PATH = "models"

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Data split ratios
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# API configuration
API_KEY = "1ec78127-e12b-4cb2-a9fb-1258e4d5622a"
COUNTRY = "DK"
