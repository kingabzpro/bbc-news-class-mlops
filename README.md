# MLOps Project Template

A comprehensive MLOps project template that integrates modern tools and frameworks for the entire machine learning lifecycle, from experiment tracking to model deployment and monitoring.

## Architecture Overview

This template implements a modern MLOps architecture that follows industry best practices:

```
+---------------------------+     +-------------------+     +-----------------+
|                           |     |                   |     |                 |
|  Data & Feature Pipeline  | --> |  Model Pipeline   | --> | Model Serving   |
|  (Feast)                  |     |  (MLflow/Prefect) |     | (FastAPI/HF)    |
|                           |     |                   |     |                 |
+---------------------------+     +-------------------+     +-----------------+
            |                              |                         |
            v                              v                         v
+---------------------------+     +-------------------+     +-----------------+
|                           |     |                   |     |                 |
|  Data/Feature Monitoring  | <-> |  Model Registry   | <-> | Model Monitoring|
|  (Evidently)              |     |  (MLflow)         |     | (Evidently)     |
|                           |     |                   |     |                 |
+---------------------------+     +-------------------+     +-----------------+
                                          |
                                          v
                                  +-------------------+
                                  |                   |
                                  |  CI/CD Pipeline   |
                                  |  (GitHub Actions) |
                                  |                   |
                                  +-------------------+
```

The architecture consists of several interconnected components that work together to provide a complete MLOps workflow:

1. **Data & Feature Engineering**: Managed through Feast feature store
2. **Model Training & Experimentation**: Tracked with MLflow and orchestrated with Prefect
3. **Model Testing**: Validation with Deepchecks
4. **Model Registry & Versioning**: Centralized in MLflow with Git LFS for storage
5. **Model Serving**: API endpoints with FastAPI and deployment to Hugging Face
6. **Monitoring**: Data and model monitoring with Evidently AI
7. **CI/CD**: Automation with GitHub Actions

## Technologies

This template integrates the following tools and frameworks:

### Package Management
- **uv**: A fast Python package installer and resolver, used for dependency management

### Experiment Tracking & Model Registry
- **MLflow**: For tracking experiments, packaging models, and managing the model registry

### Workflow Orchestration
- **Prefect**: For creating, scheduling, and monitoring production ML pipelines

### Version Control
- **Git** & **Git LFS**: For code versioning and managing large model files and datasets

### Feature Store
- **Feast**: For managing, storing, and serving machine learning features

### Model Testing
- **Deepchecks**: For validating ML models and data

### Model Serving
- **FastAPI**: For creating high-performance API endpoints for model serving
- **Hugging Face**: For model deployment and sharing

### Monitoring
- **Evidently AI**: For monitoring model performance and data drift in production

### CI/CD
- **GitHub Actions**: For continuous integration and deployment pipelines

## Project Structure

```
A-Mlops/
├── .github/                   # GitHub configuration
│   └── workflows/             # GitHub Actions workflows
├── configs/                   # Configuration files
│   ├── feast/                 # Feast feature store configuration
│   ├── deepchecks.yaml        # Deepchecks configuration
│   ├── evidently.yaml         # Evidently configuration
│   ├── huggingface.yaml       # Hugging Face configuration
│   ├── mlflow.yaml            # MLflow configuration
│   └── prefect.toml           # Prefect configuration
├── deploy/                    # Deployment configurations
├── docs/                      # Documentation
│   ├── CI_CD.md               # CI/CD documentation
│   ├── DEPLOYMENT.md          # Deployment documentation
│   ├── ENVIRONMENT.md         # Environment setup documentation
│   ├── EXPERIMENTS.md         # Experiment tracking documentation
│   ├── FEATURES.md            # Feature store documentation
│   ├── MONITORING.md          # Monitoring documentation
│   ├── SERVING.md             # Model serving documentation
│   ├── TESTING.md             # Model testing documentation
│   └── WORKFLOWS.md           # Workflow orchestration documentation
├── examples/                  # Example notebooks and scripts
│   ├── deploy_hf.py           # Example HF deployment
│   ├── feature_retrieval.py   # Example Feast feature retrieval
│   ├── monitoring_dashboard.ipynb # Example monitoring dashboard
│   ├── register_model.py      # Example model registration
│   ├── request.json           # Example API request
│   └── response.json          # Example API response
├── infra/                     # Infrastructure as code
├── scripts/                   # Utility scripts
│   ├── push_to_hf.py          # Script for pushing models to HF
│   ├── setup_env.bat          # Windows environment setup
│   └── setup_env.sh           # Linux/macOS environment setup
├── src/                       # Source code
│   ├── data/                  # Data processing code
│   ├── features/              # Feature engineering code
│   │   └── feature_repo.py    # Feast feature definitions
│   ├── models/                # Model definitions
│   ├── pipelines/             # Pipeline definitions
│   │   ├── prefect_flows/     # Prefect flow definitions
│   │   └── mlflow_example.py  # MLflow integration example
│   ├── serving/               # Model serving code
│   │   ├── app.py             # FastAPI application
│   │   └── schemas.py         # API schemas
│   └── monitoring/            # Monitoring code
│       └── monitor.py         # Evidently integration
├── tests/                     # Tests
│   ├── data_validation/       # Data validation tests
│   │   └── schema_check.py    # Data schema validation
│   ├── model_validation/      # Model validation tests
│   │   └── train_test_checks.py # Train-test validation
│   └── performance_checks/    # Performance validation tests
├── .gitattributes             # Git LFS configuration
├── .gitignore                 # Git ignore patterns
├── LICENSE                    # Project license
├── pyproject.toml             # Python project configuration
├── README.md                  # Project documentation
├── requirements-dev.txt       # Development dependencies
├── requirements-prod.txt      # Production dependencies
└── requirements-test.txt      # Testing dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- Git with Git LFS
- uv package manager

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/A-Mlops.git
   cd A-Mlops
   ```

2. Set up the environment:
   
   **On Windows:**
   ```powershell
   .\scripts\setup_env.bat
   ```
   
   **On Linux/macOS:**
   ```bash
   ./scripts/setup_env.sh
   ```

3. Activate the environment:
   
   **On Windows:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **On Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements-dev.txt
   ```

### Configuration

1. Configure MLflow:
   ```bash
   # Edit configs/mlflow.yaml to set your tracking URI and artifact store
   mlflow server --config-path configs/mlflow.yaml
   ```

2. Configure Prefect:
   ```bash
   # Set up Prefect server if needed
   prefect server start
   ```

3. Configure Feast:
   ```bash
   # Initialize Feast feature store
   cd src/features
   feast apply
   ```

### Running Example Flows

1. Start an example Prefect flow:
   ```bash
   python src/pipelines/prefect_flows/example_flow.py
   ```

2. Start the model serving API:
   ```bash
   uvicorn src.serving.app:app --reload
   ```

3. Run model monitoring:
   ```bash
   python src/monitoring/monitor.py
   ```

## Documentation

For more detailed documentation on specific components, see the documentation files in the `docs/` directory:

- [Environment Setup](docs/ENVIRONMENT.md)
- [Experiment Tracking](docs/EXPERIMENTS.md)
- [Workflow Orchestration](docs/WORKFLOWS.md)
- [Feature Store](docs/FEATURES.md)
- [Model Testing](docs/TESTING.md)
- [Model Serving](docs/SERVING.md)
- [Model Deployment](docs/DEPLOYMENT.md)
- [Model Monitoring](docs/MONITORING.md)
- [CI/CD Pipelines](docs/CI_CD.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This template is inspired by best practices in the MLOps community
- Thanks to all the open-source projects that make MLOps possible

