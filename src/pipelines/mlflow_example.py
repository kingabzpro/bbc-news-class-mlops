#!/usr/bin/env python
"""
MLflow Experiment Tracking and Model Registry Example

This script demonstrates how to use MLflow for:
1. Experiment tracking
2. Parameter and metric logging
3. Artifact storage
4. Model registration

It trains a simple scikit-learn model on synthetic data.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlflow-example")


def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file (None to use default)
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Default config path in project structure
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "configs", "mlflow.yaml")
        
    # Fallback if file doesn't exist
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using environment variables")
        config = {
            "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            "experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", "example-experiment"),
            "artifact_root": os.environ.get("MLFLOW_ARTIFACT_ROOT", "./mlruns"),
        }
        return config
    
    # Load from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def generate_synthetic_data(n_samples=1000, n_features=10, random_state=42):
    """
    Generate synthetic data for demonstration purposes.
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    
    # Generate features names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create target variable (binary classification)
    # Higher values of first 3 features increase likelihood of positive class
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    
    # Add some noise
    noise = np.random.randn(n_samples) * 0.1
    y = (y + noise > 0).astype(int)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Convert to DataFrame for better tracking
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Return datasets and feature names
    return X_train_df, X_test_df, y_train, y_test, feature_names


def create_artifacts(X_test, y_test, y_pred, y_pred_proba, feature_importance, output_dir="artifacts"):
    """
    Create artifacts like plots, reports, and feature importance.
    
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        feature_importance: Feature importance scores
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary mapping artifact paths to descriptions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    artifacts = {}
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ["Negative", "Positive"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    artifacts["confusion_matrix"] = "Confusion matrix showing prediction results"
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    artifacts["roc_curve"] = "ROC curve showing model performance"
    
    # Feature importance plot
    # Sort features by importance
    feature_importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    feature_importance_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(feature_importance_path)
    plt.close()
    artifacts["feature_importance"] = "Feature importance visualization"
    
    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    artifacts["classification_report"] = "Detailed classification metrics"
    
    # Save sample predictions
    sample_size = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    sample_predictions = pd.DataFrame({
        'true_label': y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices],
        'predicted_label': y_pred[sample_indices],
        'probability': y_pred_proba[sample_indices]
    })
    predictions_path = os.path.join(output_dir, "sample_predictions.csv")
    sample_predictions.to_csv(predictions_path, index=False)
    artifacts["sample_predictions"] = "Sample of predictions with ground truth"
    
    return artifacts


def train_and_log_model(config, args):
    """
    Train a model and log it with MLflow.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri(config.get("tracking_uri", "http://localhost:5000"))
    experiment_name = config.get("experiment_name", "example-experiment")
    
    # Create or get the experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=config.get("artifact_root")
            )
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Error setting up experiment: {e}")
        # Fallback to default experiment
        experiment_id = "0"
        logger.info("Using default experiment")
    
    # Set the active experiment
    mlflow.set_experiment(experiment_name)
    
    # Generate data
    logger.info("Generating synthetic data")
    n_samples = args.n_samples
    n_features = args.n_features
    X_train, X_test, y_train, y_test, feature_names = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        random_state=args.random_state
    )
    
    # Start an MLflow run
    run_name = f"rf_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log the run information
        logger.info(f"Started MLflow run with ID: {run.info.run_id}")
        
        # Log input data parameters
        mlflow.log_params({
            "data_type": "synthetic",
            "n_samples": n_samples,
            "n_features": n_features,
            "random_state": args.random_state,
            "test_size": 0.2,
        })
        
        # Define model parameters
        model_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
        }
        
        # Create and train the model
        logger.info("Training model with parameters: %s", model_params)
        
        # Log model parameters
        mlflow.log_params(model_params)
        
        # Create and train the model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Positive class probability
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("Model performance metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        })
        
        # Get feature importance
        feature_importance = model.feature_importances_
        
        # Create artifacts directory for visualizations and reports
        artifact_dir = "artifacts"
        if os.path.exists(artifact_dir):
            import shutil
            shutil.rmtree(artifact_dir)
        
        # Create and save artifacts (plots, reports, etc.)
        artifacts = create_artifacts(
            X_test, 
            y_test, 
            y_pred, 
            y_pred_proba, 
            feature_importance,
            output_dir=artifact_dir
        )
        
        # Log artifacts to MLflow
        for artifact_path in os.listdir(artifact_dir):
            full_path = os.path.join(artifact_dir, artifact_path)
            if os.path.isfile(full_path):
                mlflow.log_artifact(full_path)
        
        # Log the model with MLflow
        mlflow.sklearn.log_model(
            model, 
            "model", 
            registered_model_name=args.model_name if args.register_model else None
        )
        
        # Create a signature for the model (for validation when serving)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log model with a signature and input example
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=args.model_name if args.register_model else None
        )
        
        # Register the model in the model registry if requested
        if args.register_model:
            # Get the client
            client = mlflow.tracking.MlflowClient()
            
            try:
                # Check if model already exists in registry
                model_details = client.get_registered_model(args.model_name)
                logger.info(f"Model '{args.model_name}' already exists in registry")
            except mlflow.exceptions.RestException:
                logger.info(f"Model '{args.model_name}' does not exist in registry, registering...")
            
            # Register model version
            model_version = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                args.model_name
            )
            logger.info(f"Registered model version: {model_version.version}")
            
            # Set model version tags
            client.set_model_version_tag(
                name=args.model_name,
                version=model_version.version,
                key="data_samples",
                value=str(n_samples)
            )
            
            client.set_model_version_tag(
                name=args.model_name,
                version=model_version.version,
                key="accuracy",
                value=str(round(accuracy, 4))
            )
            
            client.set_model_version_tag(
                name=args.model_name,
                version=model_version.version,
                key="f1_score",
                value=str(round(f1, 4))
            )
            
            # Transition model to staging or production based on performance
            stage = "Staging"
            if args.auto_promote and accuracy >= args.promotion_threshold:
                stage = "Production"
                
                # Archive existing production models
                production_versions = client.get_latest_versions(args.model_name, stages=["Production"])
                for production_version in production_versions:
                    logger.info(f"Archiving existing production model version: {production_version.version}")
                    client.transition_model_version_stage(
                        name=args.model_name,
                        version=production_version.version,
                        stage="Archived"
                    )
            
            # Transition this model to the target stage
            logger.info(f"Transitioning model to {stage}")
            client.transition_model_version_stage(
                name=args.model_name,
                version=model_version.version,
                stage=stage
            )
            
            # Add description
            client.update_model_version(
                name=args.model_name,
                version=model_version.version,
                description=f"Model trained on {n_samples} samples with accuracy {accuracy:.4f}"
            )
            
            # Print model registry URL
            registry_url = f"{config.get('tracking_uri')}/#/models/{args.model_name}/versions/{model_version.version}"
            logger.info(f"Model registered at: {registry_url}")
        
        # Print run info
        logger.info(f"Run completed: {run.info.run_id}")
        logger.info(f"Artifacts stored at: {run.info.artifact_uri}")
        
        # Return the run ID for reference
        return run.info.run_id


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace of arguments
    """
    parser = argparse.ArgumentParser(description="MLflow Model Training and Registration Example")
    
    # Data generation arguments
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--n-features", type=int, default=10, help="Number of features to generate")
    
    # Model parameters
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of trees")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split a node")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples required at a leaf node")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    
    # MLflow configuration
    parser.add_argument("--config", type=str, help="Path to MLflow configuration file")
    parser.add_argument("--experiment-name", type=str, help="MLflow experiment name (overrides config)")
    
    # Model registry arguments
    parser.add_argument("--register-model", action="store_true", help="Register model in the MLflow Model Registry")
    parser.add_argument("--model-name", type=str, default="example_model", help="Name for the registered model")
    parser.add_argument("--auto-promote", action="store_true", help="Automatically promote model to production if performance is good")
    parser.add_argument("--promotion-threshold", type=float, default=0.95, help="Accuracy threshold for auto-promotion to production")
    
    return parser.parse_args()


def main():
    """
    Main function to run the MLflow example.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    
    # Train and log the model with MLflow
    run_id = train_and_log_model(config, args)
    
    # Print final message
    logger.info("Process completed successfully")
    logger.info(f"Run ID: {run_id}")
    
    # Provide URL to the experiment in the MLflow UI
    if "tracking_uri" in config:
        url = f"{config['tracking_uri']}/#/experiments/{mlflow.get_experiment_by_name(config['experiment_name']).experiment_id}/runs/{run_id}"
        logger.info(f"View experiment at: {url}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

