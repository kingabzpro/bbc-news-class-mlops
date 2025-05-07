"""
Model training script with MLflow tracking.
"""

import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

os.environ["JOBLIB_START_METHOD"] = "spawn"
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "mlflow_config.yaml"


def load_mlflow_config():
    """Load MLflow configuration from YAML file"""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_mlflow():
    """Set up MLflow tracking"""
    config = load_mlflow_config()

    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(
        f"MLflow experiment: {mlflow.get_experiment_by_name(config['experiment_name'])}"
    )


def create_pipeline(classifier_type="logistic"):
    """
    Create a sklearn pipeline with TF-IDF and classifier.

    Args:
        classifier_type: Type of classifier ('logistic', 'svm', or 'rf')

    Returns:
        sklearn Pipeline
    """
    if classifier_type == "logistic":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_type == "svm":
        classifier = LinearSVC(random_state=42)
    elif classifier_type == "rf":
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_model(classifier_type="logistic", tune_hyperparams=True):
    """
    Train a text classification model and track with MLflow.

    Args:
        classifier_type: Type of classifier ('logistic', 'svm', or 'rf')
        tune_hyperparams: Whether to tune hyperparameters with GridSearchCV

    Returns:
        Trained model pipeline
    """
    # Set up MLflow
    setup_mlflow()

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    X_train = train_df["title"]
    y_train = train_df["category"]
    X_val = val_df["title"]
    y_val = val_df["category"]

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Classes: {np.unique(y_train)}")

    # Create pipeline
    pipeline = create_pipeline(classifier_type)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("classifier_type", classifier_type)
        mlflow.log_param("tune_hyperparams", tune_hyperparams)

        if tune_hyperparams:
            # Define hyperparameter grid
            if classifier_type == "logistic":
                param_grid = {
                    "tfidf__max_features": [5000, 10000],
                    "tfidf__ngram_range": [(1, 1), (1, 2)],
                    "classifier__C": [0.1, 1.0, 10.0],
                }
            elif classifier_type == "svm":
                param_grid = {
                    "tfidf__max_features": [5000, 10000],
                    "tfidf__ngram_range": [(1, 1), (1, 2)],
                    "classifier__C": [0.1, 1.0, 10.0],
                }
            elif classifier_type == "rf":
                param_grid = {
                    "tfidf__max_features": [5000, 10000],
                    "tfidf__ngram_range": [(1, 1), (1, 2)],
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 10, 20],
                }

            # Log hyperparameter search space
            for param, values in param_grid.items():
                mlflow.log_param(f"grid_{param}", values)

            # Tune hyperparameters
            logger.info("Tuning hyperparameters")
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Use best model
            pipeline = grid_search.best_estimator_

            # Log best hyperparameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
        else:
            # Train model without tuning
            logger.info("Training model without hyperparameter tuning")
            pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        logger.info(f"Validation accuracy: {accuracy:.4f}")
        logger.info(f"Validation F1 score: {f1:.4f}")

        # Log classification report
        report = classification_report(y_val, y_pred)
        logger.info(f"Classification report:\n{report}")

        # Log model
        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)

        # Save model locally
        model_path = MODEL_DIR / f"news_classifier_{classifier_type}.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to {model_path}")

        return pipeline


if __name__ == "__main__":
    train_model(classifier_type="logistic", tune_hyperparams=True)
