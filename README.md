# News Classification MLOps

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLOps](https://img.shields.io/badge/MLOps-Enabled-green.svg)](https://mlops.org)

An end-to-end MLOps pipeline for training, deploying, and monitoring a news classification model using BBC articles dataset from Kaggle. This project demonstrates best practices in MLOps including model training, deployment, monitoring, and CI/CD.

## 🚀 Features

- **FastAPI Service**: Async API with multi-worker support and API key authentication
- **ML Pipeline**: Automated data processing, training, and evaluation using Prefect
- **Model Management**: Experiment tracking and model versioning with MLflow
- **Monitoring**: Real-time metrics and dashboards with Prometheus & Grafana
- **Testing**: Unit tests, integration tests, and load testing with Locust
- **Containerization**: Docker support for easy deployment
- **CI/CD**: Automated testing and deployment with GitHub Actions

## 📋 Prerequisites

- Docker and Docker Compose

## 🏗️ Project Structure

```
.
├── src/
│   ├── data/         # Data download and preprocessing
│   ├── models/       # Model training and evaluation
│   ├── api/          # FastAPI service implementation
│   └── pipelines/    # Prefect workflow definitions
├── tests/            # Unit and integration tests
├── notebooks/        # Jupyter notebooks for exploration
├── configs/          # Configuration files
└── workflows/        # CI/CD workflow definitions
```

## 🚀 Getting Started

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/kingabzpro/A-to-Z-MLOps.git
cd A-to-Z-MLOps
```

### 2. Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Update `.env` with your configuration:
   ```env
   API_KEY=your_api_key
   CACHE_TTL=3600
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

> ⚠️ **Security Note**: Never commit your `.env` file to version control.



### 3. Docker Compose

```bash
# Start all services
docker-compose up -d
```

Available services:
- FastAPI: [http://localhost:7860/docs](http://localhost:7860/docs)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (admin/admin)
- MLflow: [http://localhost:5000](http://localhost:5000)
- Prefect: [http://localhost:4200](http://localhost:4200)
- Locust: [http://localhost:8089](http://localhost:8089)


```bash
# Stop all services
docker-compose down
```


## 🔧 Troubleshooting

1. **API Connection Issues**
   - Verify API key is correctly set in `.env`
   - Check if the service is running on the correct port
   - Ensure all required environment variables are set

2. **Docker Issues**
   - Ensure Docker daemon is running
   - Check port conflicts
   - Verify Docker Compose version compatibility

## 🛠️ Tech Stack

- **Package Management**: uv
- **Data Source**: Kaggle
- **Machine Learning**: scikit-learn
- **Model Management**: MLflow
- **Workflow Orchestration**: Prefect
- **API Framework**: FastAPI
- **Load Testing**: Locust
- **Monitoring**: Prometheus & Grafana
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions

## 📝 Stats and Dashboards


### Project Strcuture
This project leverages a robust MLOps pipeline, integrating data ingestion, model training, evaluation, deployment, monitoring, and testing. The workflow is as follows:

```mermaid
graph TD
  subgraph Data_Pipeline
    A["Kaggle API (Dataset)"] --> B["Data Processing"]
    B --> C["Model Training & Evaluation (scikit-learn)"]
    C -->|"Tracked by"| D["MLflow (Tracking & Model Registry)"]
  end

  subgraph Orchestration
    E["Prefect (Workflow Orchestration)"]
    A --> E
    B --> E
    C --> E
    D --> E
  end

  subgraph API_and_Testing
    F["FastAPI (Loads Model from MLflow)"]
    F -->|"Endpoints Tested by"| G["Pytest"]
    F -->|"Stress Tested by"| H["Locust"]
  end

  D --> F

  subgraph Monitoring_and_Dashboard
    F --> I["Prometheus (Metrics)"]
    I --> J["Grafana (Visualization)"]
  end

  subgraph Deployment
    L["Docker Compose"] --> F
    L --> D
    L --> I
    L --> J
    L --> H
    L --> E
  end

  M["GitHub Actions (CI/CD, Optional)"] --> L
```

### Prefect
Prefect orchestrates the workflow, running all steps in sequence and generating detailed logs for each stage.
![Prefect Dashboard](./images/prefect.png)

### MLFlow

MLflow tracks all model training metrics and manages the model registry.
![MLFlow Experiment](./images/mlflow_1.png)

The trained model is registered in the MLflow Model Registry and loaded by the FastAPI application.
![MLFlow Model Registry](./images/mlflow_2.png)


### Machine Learning Application
The FastAPI application exposes three endpoints: `/info`, `/predict`, and `/metrics`.
![ML APP Docs](./images/api.png)

### Locust
Locust is used for stress testing the API, providing detailed statistics and real-time graphs of request performance.
![Locust Stats](./images/locust_1.png)
Stress test graph to observe what happens at each second.
![Locust Graphs](./images/locust_2.png)

### Prometheus
Prometheus collects and displays system and application metrics, enabling real-time monitoring.
![Prometheus Dashboard](./images/prometheus.png)

### Grafana
Grafana dashboards provide a comprehensive overview of the deployed model's performance and system health.
![Grafana Dashboard](./images/grafana.png)

### Docker
Docker Desktop shows all running services and their respective ports, simplifying service management.
![Docker Dashboard](./images/docker.png)