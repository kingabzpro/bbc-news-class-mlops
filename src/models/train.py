"""
Model training script with MLflow tracking.
"""

import logging
import os
import warnings
from pathlib import Path

import joblib

# Added imports for new metrics and plotting
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
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
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)

        logger.info(f"Validation accuracy: {accuracy:.4f}")
        logger.info(f"Validation F1 score (weighted): {f1:.4f}")
        logger.info(f"Validation Precision (weighted): {precision:.4f}")
        logger.info(f"Validation Recall (weighted): {recall:.4f}")

        # Log confusion matrix
        logger.info("Calculating and logging confusion matrix")
        cm = confusion_matrix(y_val, y_pred, labels=pipeline.classes_)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=pipeline.classes_,
            yticklabels=pipeline.classes_,
            ax=ax_cm,
            cmap="Blues",
        )
        ax_cm.set_xlabel("Predicted labels")
        ax_cm.set_ylabel("True labels")
        ax_cm.set_title("Confusion Matrix")
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        # Log Log Loss and ROC AUC curve if predict_proba is available
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_val)

            # Log Loss
            try:
                ll = log_loss(y_val, y_pred_proba, labels=pipeline.classes_)
                mlflow.log_metric("log_loss", ll)
                logger.info(f"Validation Log Loss: {ll:.4f}")
            except ValueError as e:
                logger.warning(f"Could not calculate Log Loss: {e}")

            # ROC AUC Curve
            logger.info("Calculating and logging ROC AUC Curve")
            fig_roc, ax_roc = plt.subplots()

            y_val_binarized = label_binarize(y_val, classes=pipeline.classes_)
            n_classes = y_val_binarized.shape[1]

            # Handle binary case where label_binarize might return 1 column but y_pred_proba has 2
            # For roc_curve, y_score should be probability of the positive class.
            # For multi-class (OVR), y_score is proba of that class.

            # Removed per-class ROC curve and AUC calculation loop
            # The plot will now primarily feature the micro-average ROC curve.

            # Compute micro-average ROC curve and ROC area
            if n_classes > 1:  # Micro-average is more meaningful for multi-class
                fpr_micro, tpr_micro, _ = roc_curve(
                    y_val_binarized.ravel(), y_pred_proba.ravel()
                )
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                ax_roc.plot(
                    fpr_micro,
                    tpr_micro,
                    label=f"micro-average ROC curve (area = {roc_auc_micro:.2f})",
                    color="deeppink",
                    linestyle=":",
                    linewidth=4,
                )
                mlflow.log_metric("roc_auc_micro", roc_auc_micro)
                logger.info(f"Validation ROC AUC (Micro): {roc_auc_micro:.4f}")
            elif n_classes == 1:  # Binary or effectively binary after binarization
                # For a single class (binary), calculate its ROC directly
                # y_val_binarized will be (n_samples, 1), y_pred_proba typically (n_samples, 2) or (n_samples, 1) if squeezed
                # We need probability of the positive class for roc_curve.
                # Assuming y_pred_proba[:, 1] is the probability of the positive class if shape is (n_samples, 2)
                # If pipeline.classes_ has 2 elements, y_val_binarized by default uses the second as positive.
                positive_class_proba = (
                    y_pred_proba[:, 1]
                    if y_pred_proba.shape[1] == 2
                    else y_pred_proba[:, 0]
                )
                fpr_binary, tpr_binary, _ = roc_curve(
                    y_val_binarized[:, 0], positive_class_proba
                )
                roc_auc_binary = auc(fpr_binary, tpr_binary)
                ax_roc.plot(
                    fpr_binary,
                    tpr_binary,
                    label=f"ROC curve (area = {roc_auc_binary:.2f})",
                    color="blue",
                    linewidth=2,
                )
                mlflow.log_metric(
                    "roc_auc", roc_auc_binary
                )  # Log as the main roc_auc for binary
                logger.info(f"Validation ROC AUC: {roc_auc_binary:.4f}")

            # Log macro-average ROC AUC score
            try:
                roc_auc_macro = roc_auc_score(
                    y_val,
                    y_pred_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=pipeline.classes_,
                )
                mlflow.log_metric("roc_auc_macro", roc_auc_macro)
                logger.info(f"Validation ROC AUC (Macro): {roc_auc_macro:.4f}")
            except ValueError as e:
                logger.warning(f"Could not calculate Macro ROC AUC: {e}")

            ax_roc.plot([0, 1], [0, 1], "k--")  # Dashed diagonal
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC)")
            ax_roc.legend(loc="lower right")

            mlflow.log_figure(fig_roc, "roc_auc_curves.png")
            plt.close(fig_roc)
        else:
            logger.warning(
                f"Classifier {classifier_type} does not support predict_proba. Skipping Log Loss and ROC AUC metrics."
            )

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
