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

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for virtual environment and package management
- Docker and Docker Compose (optional)

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
git clone https://github.com/kingabzpro/bbc-news-class-mlops.git
cd bbc-news-class-mlops

# Create and activate virtual environment
uv venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt
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
- FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
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

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
