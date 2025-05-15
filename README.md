# News Classification MLOps Project

This MLOps project provides an end-to-end pipeline for training and deploying a news classification model using the BBC articles dataset from Kaggle.

## Project Structure

- `src/`: Source code
  - `data/`: Data download and preprocessing
  - `models/`: Model training and evaluation
  - `api/`: FastAPI service for model serving
  - `pipelines/`: Prefect workflows for orchestration
- `tests/`: Unit and integration tests
- `notebooks/`: Jupyter notebooks for exploration
- `configs/`: Configuration files
- `workflows/`: CI/CD workflows

## Requirements

- Python 3.10+
- uv package manager
- Access to Kaggle API

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/kingabzpro/bbc-news-class-mlops.git
   cd bbc-news-class-mlops
   ```

2. Set up environment with uv:
   ```
   uv venv
   .venv\Scripts\activate 
   uv pip install -r requirements.txt
   ```

3. Run the training pipeline:
   ```
   python -m src.pipelines.pipeline
   ```

4. Start the API server:
   ```
   python -m src.api.main
   ```

## Usage Instructions

### 1. Environment Setup
- Install [uv](https://github.com/astral-sh/uv):
  ```sh
  pip install uv
  ```
- Create a virtual environment and install dependencies:
  ```sh
  uv venv
  uv pip install -r requirements.txt
  ```

### 2. Kaggle API Setup
- Create a Kaggle account and get your API token from https://www.kaggle.com/settings/account.
- Place your `kaggle.json` in `%USERPROFILE%\.kaggle\` (Windows) or `~/.kaggle/` (Linux/Mac).

### 3. Download and Preprocess Data
- Download the dataset:
  ```sh
  python -m src.data.download
  ```
- Preprocess the dataset:
  ```sh
  python -m src.data.preprocess
  ```

### 4. Run the MLOps Pipeline
- Orchestrate the full workflow (download, preprocess, train, evaluate):
  ```sh
  python -m src.pipelines.pipeline
  ```

### 5. Serve the Model API
- Start the FastAPI server:
  ```sh
  python -m src.api.main
  ```
- Access the docs at: http://localhost:8000/docs
- The API is protected. To authenticate, you need to set an `API_KEY` environment variable before starting the server (e.g., in a `.env` file).
- When making requests to the `/predict` endpoint, include the API key in the `X-API-Key` header.

  Example request using `curl`:
  ```sh
  curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -H "X-API-Key: YOUR_API_KEY" \
       -d '{"title": "New research on climate change"}'
  ```

### 5a. New Dockerfile: Build and Run with Docker

You can also run the API using Docker. The Dockerfile will run tests (but not fail the build if they fail) and then launch the FastAPI app on port 7860.

1. **Build the Docker image:**
   ```sh
   docker build -t news_class_app .
   ```

2. **Run the Docker container:**
   - You can set the `API_KEY` using an `.env` file or with the `--env` flag.
   - Example using an `.env` file:
     ```sh
     docker run --env-file .env -p 7860:7860 news_class_app
     ```
   - Or set the variable directly:
     ```sh
     docker run -e API_KEY=your_api_key -p 7860:7860 news_class_app
     ```

3. **Access the API docs:**
   - Open http://localhost:7860/docs in your browser.

### 6. Run Tests
- Run all tests:
  ```sh
  pytest
  ```

### 7. MLflow Tracking
- MLflow UI (after running training):
  ```sh
  mlflow ui
  ```
- Open http://127.0.0.1:5000 to view experiments.

### 8. CI/CD
- GitHub Actions will automatically lint, test, and check pipeline/API on push or PR to `main`.

---

### 8. FastAPI Monitoring with Prometheus
To enable Prometheus to scrape metrics from your FastAPI application, you need to instrument your app to expose a metrics endpoint (commonly `/metrics`). This typically involves using a library like `prometheus-fastapi-instrumentator`.
1. Go to the official Prometheus downloads page: https://prometheus.io/download/ 
2. Extract the contents of the downloaded archive into this C:\Prometheus folder.
3. Open the `prometheus.yml` file located in your Prometheus installation directory with a text editor and add the following configuration:
```yml
  - job_name: 'fastapi-app'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8000'] 
```
4. Start the Powershell in the same folder and run the following command:
```powershell
prometheus.exe --config.file=prometheus.yml --web.listen-address=":9090"
```
5. Open your web browser and go to http://localhost:9090. You should see the Prometheus UI.

## MLOps Components

- **Package Management**: uv
- **Experiment Tracking**: MLflow
- **Workflow Orchestration**: Prefect
- **Model Serving**: FastAPI
- **Data Management**: Kaggle
- **Monitoring**: Prometheus
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
