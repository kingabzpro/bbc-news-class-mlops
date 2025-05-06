#!/usr/bin/env python
"""
Prefect Workflow Example

This script demonstrates how to use Prefect for workflow orchestration:
1. Setting up a flow with tasks
2. Configuring task dependencies
3. Handling failures
4. Integrating with MLflow
5. Reading configuration from Prefect config

The example simulates a complete ML pipeline with the following steps:
- Data extraction
- Data validation
- Feature engineering
- Model training
- Model evaluation
- Model deployment (conditional)
"""

import os
import sys
import time
import uuid
import toml
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import NotFittedError

# Import Prefect
from prefect import flow, task, get_run_logger
from prefect.context import get_run_context
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from prefect.blocks.system import JSON, String
from prefect.flows import FlowRun
from prefect.artifacts import create_markdown_artifact
import prefect.context
from prefect.exceptions import MissingContextError


# Add project root to path
# This allows importing from the project's modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_config():
    """
    Load Prefect configuration from file or environment variables.
    
    Returns:
        dict: Configuration dictionary
    """
    logger = get_run_logger()
    
    # Check for config file in standard locations
    config_paths = [
        os.path.join(project_root, "configs", "prefect.toml"),
        os.environ.get("PREFECT_CONFIG_PATH", "")
    ]
    
    config = {}
    for config_path in config_paths:
        if config_path and os.path.exists(config_path):
            try:
                logger.info(f"Loading config from {config_path}")
                with open(config_path, "r") as f:
                    config = toml.load(f)
                break
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
    
    # Apply environment variables as overrides
    if "PREFECT_API_URL" in os.environ:
        if "prefect" not in config:
            config["prefect"] = {}
        if "api" not in config["prefect"]:
            config["prefect"]["api"] = {}
        config["prefect"]["api"]["url"] = os.environ["PREFECT_API_URL"]
    
    if "MLFLOW_TRACKING_URI" in os.environ:
        if "integrations" not in config:
            config["integrations"] = {}
        if "mlflow" not in config["integrations"]:
            config["integrations"]["mlflow"] = {}
        config["integrations"]["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    
    return config


@task(name="extract_data", 
      description="Extract data from source",
      retries=3,
      retry_delay_seconds=10,
      tags=["data", "extract"])
def extract_data(n_samples: int = 1000, 
                n_features: int = 20, 
                test_size: float = 0.2, 
                random_state: int = 42,
                failure_probability: float = 0.0) -> Dict[str, Any]:
    """
    Extract data from source (simulated with synthetic data).
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        test_size: Test set size as a fraction
        random_state: Random seed for reproducibility
        failure_probability: Probability of simulated failure (for testing error handling)
        
    Returns:
        Dictionary with training and test data
    """
    logger = get_run_logger()
    logger.info(f"Extracting data: {n_samples} samples with {n_features} features")
    
    # Simulate potential failures (useful for testing error handling)
    if random.random() < failure_probability:
        raise ValueError("Simulated extraction failure")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=int(n_features * 0.8),  # 80% of features are informative
        n_redundant=int(n_features * 0.1),    # 10% of features are redundant
        n_repeated=0,
        n_classes=2,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Convert to DataFrame for better handling
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Extracted {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Return both datasets
    return {
        "train_data": train_df,
        "test_data": test_df,
        "feature_names": feature_names,
    }


@task(name="validate_data",
      description="Validate data quality",
      tags=["data", "validation"])
def validate_data(data: Dict[str, Any], 
                 min_samples: int = 100, 
                 max_missing_pct: float = 0.1) -> Dict[str, Any]:
    """
    Validate data quality and log validation results.
    
    Args:
        data: Dictionary with training and test data
        min_samples: Minimum number of samples required
        max_missing_pct: Maximum percentage of missing values allowed
        
    Returns:
        Validated data with additional validation info
    """
    logger = get_run_logger()
    
    # Extract data
    train_df = data["train_data"]
    test_df = data["test_data"]
    
    # Initialize validation results
    validation_results = {
        "passed": True,
        "checks": [],
    }
    
    # Check 1: Minimum number of samples
    train_samples_check = len(train_df) >= min_samples
    validation_results["checks"].append({
        "name": "min_train_samples",
        "passed": train_samples_check,
        "value": len(train_df),
        "threshold": min_samples
    })
    
    if not train_samples_check:
        validation_results["passed"] = False
        logger.warning(f"Training data has only {len(train_df)} samples, minimum required is {min_samples}")
    
    # Check 2: Missing values
    missing_train = train_df.isnull().mean().max()
    missing_check = missing_train <= max_missing_pct
    validation_results["checks"].append({
        "name": "max_missing_values",
        "passed": missing_check,
        "value": missing_train,
        "threshold": max_missing_pct
    })
    
    if not missing_check:
        validation_results["passed"] = False
        logger.warning(f"Training data has {missing_train:.2%} missing values, maximum allowed is {max_missing_pct:.2%}")
    
    # Check 3: Target distribution (check for class imbalance)
    train_target_counts = train_df["target"].value_counts(normalize=True)
    min_class_pct = train_target_counts.min()
    class_balance_check = min_class_pct >= 0.1  # Ensure each class is at least 10% of data
    validation_results["checks"].append({
        "name": "class_balance",
        "passed": class_balance_check,
        "value": min_class_pct,
        "threshold": 0.1
    })
    
    if not class_balance_check:
        validation_results["passed"] = False
        logger.warning(f"Training data has class imbalance. Minority class is only {min_class_pct:.2%} of data")
    
    # Log validation results
    logger.info(f"Data validation {'passed' if validation_results['passed'] else 'failed'}")
    for check in validation_results["checks"]:
        status = "✓" if check["passed"] else "✗"
        logger.info(f"  {status} {check['name']}: {check['value']} (threshold: {check['threshold']})")
    
    # Create a Markdown artifact with validation results
    markdown_content = f"""
    # Data Validation Results
    
    Status: **{'PASSED' if validation_results['passed'] else 'FAILED'}**
    
    ## Validation Checks
    
    | Check | Threshold | Value | Status |
    |-------|-----------|-------|--------|
    """
    
    for check in validation_results["checks"]:
        status = "✓" if check["passed"] else "✗"
        markdown_content += f"| {check['name']} | {check['threshold']} | {check['value']} | {status} |\n"
    
    create_markdown_artifact(
        key="data-validation-results",
        markdown=markdown_content,
        description="Results of data validation checks"
    )
    
    # Add validation results to the data dictionary
    data["validation_results"] = validation_results
    
    return data


@task(name="engineer_features",
      description="Perform feature engineering",
      tags=["data", "features"])
def engineer_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform feature engineering on the datasets.
    
    Args:
        data: Dictionary with training and test data
        
    Returns:
        Data with engineered features
    """
    logger = get_run_logger()
    
    # Extract data
    train_df = data["train_data"]
    test_df = data["test_data"]
    feature_names = data["feature_names"]
    
    # Add engineered features to both train and test datasets
    
    # Feature 1: Interaction terms between first 5 features
    logger.info("Creating interaction features")
    for i in range(min(5, len(feature_names))):
        for j in range(i+1, min(5, len(feature_names))):
            feat_i = feature_names[i]
            feat_j = feature_names[j]
            interaction_name = f"interaction_{i}_{j}"
            
            # Create interaction feature (product of features)
            train_df[interaction_name] = train_df[feat_i] * train_df[feat_j]
            test_df[interaction_name] = test_df[feat_i] * test_df[feat_j]
    
    # Feature 2: Polynomial features for first 3 features
    logger.info("Creating polynomial features")
    for i in range(min(3, len(feature_names))):
        feat = feature_names[i]
        poly_name = f"{feat}_squared"
        
        # Create squared feature
        train_df[poly_name] = train_df[feat] ** 2
        test_df[poly_name] = test_df[feat] ** 2
    
    # Feature 3: Aggregate features (mean of all features)
    logger.info("Creating aggregate features")
    train_df["mean_of_features"] = train_df[feature_names].mean(axis=1)
    test_df["mean_of_features"] = test_df[feature_names].mean(axis=1)
    
    # Feature 4: Add feature magnitudes
    logger.info("Creating magnitude features")
    train_df["feature_magnitude"] = np.sqrt((train_df[feature_names] ** 2).sum(axis=1))
    test_df["feature_magnitude"] = np.sqrt((test_df[feature_names] ** 2).sum(axis=1))
    
    # Update feature names with the new engineered features
    new_feature_names = list(train_df.columns.drop("target"))
    
    # Log feature engineering results
    original_feature_count = len(feature_names)
    new_feature_count = len(new_feature_names)
    logger.info(f"Feature engineering complete: {original_feature_count} original features, {new_feature_count} after engineering")
    
    # Update the data dictionary
    data["train_data"] = train_df
    data["test_data"] = test_df
    data["feature_names"] = new_feature_names
    data["original_feature_names"] = feature_names
    
    return data


@task(name="train_model",
      description="Train a machine learning model",
      timeout_seconds=600,  # 10 minutes timeout
      tags=["model", "training"])
def train_model(data: Dict[str, Any], 
               model_type: str = "random_forest",
               n_estimators: int = 100,
               max_depth: int = None,
               random_state: int = 42) -> Dict[str, Any]:
    """
    Train a machine learning model.
    
    Args:
        data: Dictionary with training and test data
        model_type: Type of model to train ("random_forest", etc.)
        n_estimators: Number of estimators for ensemble models
        max_depth: Maximum depth for tree models
        random_state: Random seed for reproducibility
        
    Returns:
        Data with trained model
    """
    logger = get_run_logger()
    
    # Extract data
    train_df = data["train_data"]
    feature_names = data["feature_names"]
    
    # Separate features and target
    X_train = train_df[feature_names]
    y_train = train_df["target"]
    
    # Set up MLflow tracking
    try:
        # Try to get config from the context
        config = prefect.context.get_run_context().flow_run.parameters.get("config", {})
        mlflow_tracking_uri = (
            config.get("integrations", {})
                .get("mlflow", {})
                .get("tracking_uri", "http://localhost:5000")
        )
    except MissingContextError:
        # Fallback if not in a flow run
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = "prefect-integration"
    mlflow.set_experiment(experiment_name)
    
    # Start an MLflow run
    with mlflow.start_run(run_name=f"prefect_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Log the MLflow run ID for reference
        run_id = run.info.run_id
        logger.info(f"Started MLflow run with ID: {run_id}")
        
        # Log training parameters
        mlflow.log_params({
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "feature_count": len(feature_names),
            "training_samples": len(X_train)
        })
        
        # Train the model based on the specified model type
        logger.info(f"Training {model_type} model with {n_estimators} estimators")
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            # Fallback to RandomForest with warning
            logger.warning(f"Model type '{model_type}' not supported, using random_forest instead")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        
        # Train the model (time it for logging)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Log training time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Log feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Log as artifact
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            # Log top 10 features as parameters
            top_features = feature_importance.head(10)
            for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
                mlflow.log_param(f"top_feature_{i+1}", f"{feature} ({importance:.4f})")
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Update data dictionary with model and run info
        data["model"] = model
        data["mlflow_run_id"] = run_id
        data["training_time"] = training_time
    
    return data


@task(name="evaluate_model",
      description="Evaluate model performance",
      tags=["model", "evaluation"])
def evaluate_model(data: Dict[str, Any], 
                  metric_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.
    
    Args:
        data: Dictionary with model and test data
        metric_thresholds: Dictionary of metric thresholds for acceptance
        
    Returns:
        Data with evaluation metrics
    """
    logger = get_run_logger()
    
    # Set default thresholds if not provided
    if metric_thresholds is None:
        metric_thresholds = {
            "accuracy": 0.7,
            "f1": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "roc_auc": 0.75
        }
    
    # Extract data and model
    test_df = data["test_data"]
    feature_names = data["feature_names"]
    model = data["model"]
    mlflow_run_id = data.get("mlflow_run_id")
    
    # Separate features and target
    X_test = test_df[feature_names]
    y_test = test_df["target"]
    
    # Check if model exists and is trained
    if model is None:
        raise ValueError("Model is missing from data dictionary")
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For metrics like AUC, we need probability scores
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Positive class probability
            has_proba = True
        except (AttributeError, NotFittedError):
            logger.warning("Model doesn't support predict_proba, skipping probability-based metrics")
            y_pred_proba = None
            has_proba = False
        
        # Calculate evaluation metrics
        evaluation_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        # Add ROC AUC if probabilities are available
        if has_proba:
            evaluation_metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        
        # Log evaluation metrics
        logger.info("Model evaluation results:")
        for metric, value in evaluation_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Check if metrics meet thresholds
        metrics_passed = True
        failing_metrics = []
        
        for metric, threshold in metric_thresholds.items():
            if metric in evaluation_metrics:
                if evaluation_metrics[metric] < threshold:
                    metrics_passed = False
                    failing_metrics.append(f"{metric} ({evaluation_metrics[metric]:.4f} < {threshold})")
        
        if metrics_passed:
            logger.info("Model performance meets all thresholds")
        else:
            logger.warning(f"Model performance fails to meet thresholds for: {', '.join(failing_metrics)}")
        
        # Add evaluation results to data
        data["evaluation_metrics"] = evaluation_metrics
        data["metrics_passed"] = metrics_passed
        data["failing_metrics"] = failing_metrics
        
        # Update metrics in the existing MLflow run if available
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                for metric, value in evaluation_metrics.items():
                    mlflow.log_metric(f"test_{metric}", value)
                mlflow.log_param("metrics_passed", metrics_passed)
        
        # Create a report artifact with evaluation results
        create_markdown_artifact(
            key="model-evaluation-report",
            markdown=f"""
            # Model Evaluation Report
            
            ## Performance Metrics
            
            | Metric | Value | Threshold | Status |
            |--------|-------|-----------|--------|
            {chr(10).join([f"| {metric} | {value:.4f} | {metric_thresholds.get(metric, 'N/A')} | {'✓' if value >= metric_thresholds.get(metric, 0) else '✗'} |" for metric, value in evaluation_metrics.items()])}
            
            ## Summary
            
            The model **{'PASSED' if metrics_passed else 'FAILED'}** the evaluation criteria.
            
            {f"Failing metrics: {', '.join(failing_metrics)}" if failing_metrics else "All metrics passed the thresholds."}
            """,
            description="Model evaluation metrics and threshold validation"
        )
        
        return data
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        # Update the data dictionary with the error
        data["evaluation_error"] = str(e)
        # Still return the data so the workflow can continue
        return data


@task(name="deploy_model",
      description="Deploy the model if it meets performance criteria",
      tags=["model", "deployment"])
def deploy_model(data: Dict[str, Any], 
                force_deploy: bool = False) -> Dict[str, Any]:
    """
    Deploy the model if it passes performance criteria.
    Only runs if the model meets the performance requirements or if forced.
    
    Args:
        data: Dictionary with model and evaluation results
        force_deploy: Whether to force deployment regardless of metrics
        
    Returns:
        Updated data dictionary with deployment info
    """
    logger = get_run_logger()
    
    # Check if model should be deployed
    model = data.get("model")
    metrics_passed = data.get("metrics_passed", False)
    evaluation_metrics = data.get("evaluation_metrics", {})
    mlflow_run_id = data.get("mlflow_run_id")
    
    if model is None:
        raise ValueError("No model found in data dictionary")
    
    # Determine if we should deploy
    should_deploy = force_deploy or metrics_passed
    
    if should_deploy:
        logger.info(f"Deploying model (force_deploy={force_deploy}, metrics_passed={metrics_passed})")
        
        # Simulate deployment process
        logger.info("Preparing model for deployment...")
        time.sleep(2)  # Simulate deployment preparation
        
        # Generate a model version ID
        model_version = f"model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Log deployment to MLflow if run ID is available
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_param("deployed", True)
                mlflow.log_param("deployment_version", model_version)
                mlflow.log_param("deployment_time", datetime.now().isoformat())
        
        # Simulate model saving
        logger.info(f"Saving model artifacts for version {model_version}...")
        time.sleep(1)  # Simulate saving
        
        # Simulate staging/production deployment
        environment = "production" if metrics_passed else "staging"
        logger.info(f"Deploying model to {environment} environment...")
        time.sleep(2)  # Simulate deployment
        
        # Create deployment record
        deployment_info = {
            "model_version": model_version,
            "environment": environment,
            "deployment_time": datetime.now().isoformat(),
            "metrics": evaluation_metrics,
            "deployed_by": "prefect-workflow"
        }
        
        # Add deployment info to data
        data["deployment"] = deployment_info
        
        # Create deployment artifact
        create_markdown_artifact(
            key="model-deployment-record",
            markdown=f"""
            # Model Deployment Record
            
            ## Deployment Details
            
            - **Model Version**: {model_version}
            - **Environment**: {environment}
            - **Deployment Time**: {deployment_info['deployment_time']}
            - **Deployed By**: {deployment_info['deployed_by']}
            
            ## Model Performance
            
            | Metric | Value |
            |--------|-------|
            {chr(10).join([f"| {metric} | {value:.4f} |" for metric, value in evaluation_metrics.items()])}
            
            ## Next Steps
            
            1. Monitor model performance in production
            2. Set up A/B testing if needed
            3. Schedule regular retraining
            """,
            description="Record of model deployment details"
        )
        
        logger.info(f"Model {model_version} successfully deployed to {environment}")
    else:
        logger.info("Model does not meet performance criteria and will not be deployed")
        logger.info(f"Failing metrics: {', '.join(data.get('failing_metrics', ['unknown']))}")
        
        # If there's an MLflow run, log that deployment was skipped
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_param("deployed", False)
                mlflow.log_param("deployment_skipped_reason", "Failed metrics thresholds")
    
    return data


@flow(name="ml_training_pipeline",
      description="End-to-end ML training and deployment workflow",
      task_runner=ConcurrentTaskRunner())
def ml_training_pipeline(
    n_samples: int = 1000,
    n_features: int = 20,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples: int = 100,
    failure_probability: float = 0.0,
    force_deploy: bool = False,
    metric_thresholds: Dict[str, float] = None
):
    """
    End-to-end ML training pipeline orchestrated with Prefect.
    
    This flow demonstrates a complete ML workflow:
    1. Extract data
    2. Validate data quality 
    3. Engineer features
    4. Train model
    5. Evaluate model
    6. Deploy model (if it meets criteria)
    
    Args:
        n_samples: Number of samples in synthetic dataset
        n_features: Number of features in synthetic dataset
        model_type: Type of model to train
        n_estimators: Number of estimators for ensemble models
        max_depth: Maximum depth for tree models
        min_samples: Minimum samples required for validation
        failure_probability: Probability to simulate a failure (for testing)
        force_deploy: Whether to force deployment regardless of metrics
        metric_thresholds: Performance metric thresholds for deployment
    
    Returns:
        Results dictionary with pipeline artifacts
    """
    # Load configuration
    config = load_config()
    
    # Get logger
    logger = get_run_logger()
    logger.info("Starting ML training pipeline")
    logger.info(f"Configuration loaded: {config.keys()}")
    
    # Set default metric thresholds if not provided
    if metric_thresholds is None:
        metric_thresholds = {
            "accuracy": 0.8,
            "f1": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "roc_auc": 0.75
        }
    
    # 1. Extract data
    logger.info("Step 1: Extracting data")
    data = extract_data(
        n_samples=n_samples,
        n_features=n_features,
        failure_probability=failure_probability
    )
    
    # 2. Validate data
    logger.info("Step 2: Validating data")
    data = validate_data(
        data=data,
        min_samples=min_samples
    )
    
    # Check if validation passed before continuing
    if not data["validation_results"]["passed"]:
        logger.warning("Data validation failed, but continuing with pipeline for demonstration")
    
    # 3. Engineer features
    logger.info("Step 3: Performing feature engineering")
    data = engineer_features(data=data)
    
    # 4. Train model
    logger.info("Step 4: Training model")
    data = train_model(
        data=data,
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # 5. Evaluate model
    logger.info("Step 5: Evaluating model")
    data = evaluate_model(
        data=data,
        metric_thresholds=metric_thresholds
    )
    
    # 6. Deploy model (conditional)
    logger.info("Step 6: Deploying model (if performance criteria met)")
    data = deploy_model(
        data=data,
        force_deploy=force_deploy
    )
    
    # Log completion
    deployment_status = "Deployed" if data.get("deployment") else "Not deployed"
    logger.info(f"ML pipeline complete! Model status: {deployment_status}")
    
    # Return results dictionary
    return data


# Helper function to parse command line arguments
def parse_args():
    """Parse command line arguments for the flow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Training Pipeline with Prefect")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples in dataset")
    parser.add_argument("--n-features", type=int, default=20, help="Number of features in dataset")
    parser.add_argument("--model-type", type=str, default="random_forest", help="Type of model to train")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators for ensemble models")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth for tree models")
    parser.add_argument("--failure-probability", type=float, default=0.0, help="Probability to simulate extraction failure")
    parser.add_argument("--force-deploy", action="store_true", help="Force model deployment regardless of metrics")
    parser.add_argument("--min-accuracy", type=float, default=0.8, help="Minimum accuracy threshold")
    parser.add_argument("--min-f1", type=float, default=0.7, help="Minimum F1 score threshold")
    
    return parser.parse_args()


# Main entry point
def main():
    """Main entry point to run the flow directly."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up metric thresholds based on arguments
    metric_thresholds = {
        "accuracy": args.min_accuracy,
        "f1": args.min_f1,
        "precision": 0.7,
        "recall": 0.7,
        "roc_auc": 0.75
    }
    
    # Print banner
    print("\n" + "=" * 50)
    print(" Prefect ML Training Pipeline Example ")
    print("=" * 50 + "\n")
    
    # Run the flow
    result = ml_training_pipeline(
        n_samples=args.n_samples,
        n_features=args.n_features,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        failure_probability=args.failure_probability,
        force_deploy=args.force_deploy,
        metric_thresholds=metric_thresholds
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print(" Pipeline Execution Summary ")
    print("=" * 50)
    
    # Extract key information
    model_type = args.model_type
    metrics = result.get("evaluation_metrics", {})
    deployment = result.get("deployment", {})
    
    # Print metrics
    print(f"\nModel Type: {model_type}")
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Print deployment status
    if deployment:
        print(f"\nDeployment Status: Deployed to {deployment.get('environment', 'unknown')}")
        print(f"Deployment Version: {deployment.get('model_version', 'unknown')}")
        print(f"Deployment Time: {deployment.get('deployment_time', 'unknown')}")
    else:
        print("\nDeployment Status: Not deployed")
        failing_metrics = result.get("failing_metrics", [])
        if failing_metrics:
            print(f"Failed Metrics: {', '.join(failing_metrics)}")
    
    print("\n" + "=" * 50 + "\n")
    
    return 0


# Run the script directly
if __name__ == "__main__":
    import sys
    sys.exit(main())

