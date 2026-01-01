# CPU Usage Prediction — MLOps Pipeline with DVC & Azure ML

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#) [![DVC](https://img.shields.io/badge/dvc-enabled-brightgreen)](#) [![Azure](https://img.shields.io/badge/azure-ml-blueviolet)](#)

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach & Models](#approach--models)
- [Features & Deliverables](#features--deliverables)
- [Project Structure](#project-structure)
- [Getting Started (Local)](#getting-started-local)
- [DVC Workflow](#dvc-workflow)
- [Azure ML Deployment](#azure-ml-deployment)
- [Results & Evaluation](#results--evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Project Overview
This repository contains an end-to-end MLOps demonstration for predicting CPU usage in cloud-native environments. It covers local model development and reproducibility using DVC, experiment tracking, and cloud deployment using Azure Machine Learning (AutoML + managed online endpoints). The goal is to produce reliable, reproducible predictions of CPU usage given workload and resource configuration parameters.

## Problem Statement
Over- or under-provisioning CPU resources in cloud and containerized environments leads to either degraded performance or wasted cost. This project builds and deploys ML models to predict CPU usage from workload and resource configuration features so orchestration systems or operators can make informed allocation decisions.

## Dataset
- Type: Tabular (CSV)
- Approx. size: 72 MB
- Key features:
  - cpu_request
  - mem_request
  - cpu_limit
  - mem_limit
  - runtime_minutes
  - controller_kind
- Target:
  - cpu_usage

(Place the canonical dataset in `data/data.csv`. Large data objects are tracked via DVC.)

## Approach & Models
- Data preprocessing and feature engineering using pandas and scikit-learn pipelines
- Models evaluated:
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
  - Linear Regression
  - Azure AutoML candidates (LightGBM, XGBoost, ElasticNet, etc.)
- Core metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)

## Features & Deliverables
- Reproducible training pipeline orchestrated by DVC
- Experiment tracking with DVC experiments and metrics
- Model registry and automated cloud training using Azure ML AutoML
- Real-time model deployment as an Azure Managed Online Endpoint (REST API)
- Simple dashboard to query the deployed model and visualize predictions

## Project Structure
cpu-usage-prediction/
- data/                 — Raw and processed datasets (DVC-tracked)
  - data.csv
- src/                  — Source code
  - train.py
  - visualize.py
  - report.py
- models/               — Trained model artifacts (DVC / registry)
- metrics/              — Saved evaluation metrics
- plots/                — Generated visualizations
- dvc.yaml              — DVC pipeline definition
- params.yaml           — Pipeline and model parameters
- requirements.txt      — Python dependencies
- README.md
- .gitignore

## Getting Started (Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/Rameshkn04/cpu-usage-prediction.git
   cd cpu-usage-prediction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Initialize DVC (if not already initialized) and pull data:
   ```bash
   dvc init          # only if starting from scratch locally
   dvc pull          # fetch DVC-tracked data and model artifacts
   ```
5. Reproduce the pipeline:
   ```bash
   dvc repro
   ```
6. View metrics and experiments:
   ```bash
   dvc metrics show
   dvc exp show
   dvc metrics diff
   ```

## DVC Workflow
- Use `dvc add data/data.csv` to version large datasets.
- Define pipeline stages in `dvc.yaml` (preprocess, train, evaluate, export).
- Run experiments with `dvc exp run` and compare experiments with `dvc exp show`.
- Store and track metrics (MAE, RMSE, R²) in the metrics file(s) configured in DVC.

Useful resources:
- DVC documentation: https://dvc.org/doc

## Azure ML Deployment (High-Level)
1. Create an Azure ML Workspace and configure authentication (CLI or SDK).
2. Register dataset and compute resources in Azure ML Studio.
3. Launch an AutoML regression experiment to identify the best model.
4. Register the selected model into the workspace model registry.
5. Deploy as a Managed Online Endpoint (real-time REST API) or as a batch endpoint.
6. Test the endpoint with JSON payloads from the dashboard or via curl/HTTP client.

Useful resources:
- Azure ML docs: https://learn.microsoft.com/azure/machine-learning/

## Results & Evaluation
- Locally, Random Forest delivered the best trade-off between accuracy and explainability.
- Azure AutoML produced competitive models (e.g., LightGBM/XGBoost) with improved metrics in certain experiments.
- Evaluation artifacts (plots, metrics) are stored under `plots/` and `metrics/`.

Suggested visualizations:
- Predicted vs Actual
- Feature importance
- Residual / Error distribution

## Contributing
Contributions, issues, and feature requests are welcome.
- Fork the project
- Create a feature branch: `git checkout -b feature/your-feature`
- Commit your changes and push
- Open a pull request with a clear description and reproducible steps

Please follow the code style in the repository and include unit tests for new functionality where applicable.

## License
This project is provided under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
Ramesh K N  
Artificial Intelligence & Machine Learning  
CMR Institute of Technology  
GitHub: [ramesh0405](https://github.com/ramesh0405)
