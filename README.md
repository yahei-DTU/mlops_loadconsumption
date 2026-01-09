# mlops_loadconsumption

This project focuses on forecasting hourly electricity load consumption for the following day using deep learning methods. Accurate short-term load forecasting plays a critical role in the operation and management of modern power systems, enabling efficient scheduling, optimized energy trading, and improved grid stability. The main objective of this project is to develop a robust predictive model capable of capturing temporal dependencies and nonlinear patterns in electricity demand.

The model developed for this purpose is a one-dimensional Convolutional Neural Network (CNN1D), a deep learning architecture effective in extracting local temporal features from sequential data. The CNN1D model is designed to learn correlations in electricity consumption data across multiple time intervals, capturing the influence of recent load behavior on future demand. Compared to traditional statistical and regression-based models, CNN1D offers a more flexible framework that can adapt to complex variations in load profiles caused by weather conditions, calendar effects, and human activity patterns.

The dataset used in this project is primarily sourced from publicly available platforms, including [Energinet](https://www.energidataservice.dk/), Denmark’s national transmission system operator, and [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), which provides standardized European electricity market and system data. These sources supply essential features such as historical load consumption, temperature, wind and solar generation, and calendar-based variables (day of week, holidays, and seasonality).

The project workflow involves data preprocessing, feature engineering, model development, training, and evaluation. Model performance is assessed using standard forecasting error metrics to ensure practical relevance. The ultimate goal is to deploy a reliable and scalable load forecasting framework through an API, so it can support energy system planning, grid stabilization, and integration of renewable energy sources.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).