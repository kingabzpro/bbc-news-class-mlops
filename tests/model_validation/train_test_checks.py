"""
Deepchecks Model Validation Example

This script demonstrates how to use Deepchecks for model validation by comparing
training and test datasets, checking for data integrity, and validating model performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

# Deepchecks imports
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    TrainTestFeatureDrift,
    TrainTestLabelDrift,
    FeatureImportance,
    ModelInferenceTime,
    ConfusionMatrixReport,
    RegressionSystematicError,
    RegressionErrorDistribution,
    ClassificationSystematicError,
    ClassImbalance,
    MultivariateDrift,
    IdentifierLeakage,
    DataDuplicates,
    MixedNulls,
    StringMismatch,
    CategoryMismatch,
    DateTrainTestLeakageDuplicates,
    TrainTestSamplesMix
)
from deepchecks.tabular.suites import (
    model_evaluation,
    train_test_validation,
    data_integrity,
    full_suite
)
import mlflow


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and prepare example data for demonstration.
    
    Returns:
        Tuple containing train and test data splits (X_train, X_test, y_train, y_test)
    """
    # This is a placeholder. In a real-world scenario, you would load your actual datasets
    try:
        # Try loading from local file or data lake
        # df = pd.read_csv("path/to/data.csv")
        
        # For demonstration, we'll generate synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(5, 2, n_samples),
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature_4': np.random.uniform(0, 100, n_samples),
            'date_col': pd.date_range(start='2022-01-01', periods=n_samples, freq='D'),
            'id_column': np.arange(1000, 1000 + n_samples)
        })
        
        # Generate target - binary classification for this example
        y = (0.2 * X['feature_1'] + 0.3 * X['feature_2'] + 0.1 * X['feature_4'] + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Intentionally introduce some issues in test set for demonstration
        # - Add some drift to feature_2
        X_test['feature_2'] = X_test['feature_2'] * 1.2
        
        # - Add some missing values
        mask = np.random.random(X_test.shape[0]) < 0.05
        X_test.loc[mask, 'feature_1'] = np.nan
        
        # - Add some duplicates
        duplicate_indices = np.random.choice(X_test.index, size=10, replace=False)
        duplicated_rows = X_test.loc[duplicate_indices]
        X_test = pd.concat([X_test, duplicated_rows], ignore_index=True)
        y_test = pd.concat([y_test, y_test.loc[duplicate_indices]], ignore_index=True)
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a random forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    # For categorical features, we would normally use one-hot encoding
    # For this example, we'll drop the categorical column for simplicity
    X_train_model = X_train.copy()
    categorical_cols = X_train_model.select_dtypes(include=['object']).columns
    X_train_model = X_train_model.drop(columns=categorical_cols.tolist() + ['date_col', 'id_column'])
    
    with mlflow.start_run(run_name="model_for_deepchecks_validation"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_model, y_train)
        
        # Log model parameters
        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
    
    return model


def prepare_deepchecks_datasets(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> Tuple[Dataset, Dataset]:
    """
    Prepare Deepchecks datasets from pandas DataFrames.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        Tuple of (train_dataset, test_dataset) for Deepchecks
    """
    # Define column types for Deepchecks
    cat_features = ['feature_3']
    date_features = ['date_col']
    
    # Create Deepchecks datasets
    train_dataset = Dataset(
        df=X_train.copy(),
        label=y_train.copy(),
        cat_features=cat_features,
        datetime_features=date_features,
        index_name='id_column'
    )
    
    test_dataset = Dataset(
        df=X_test.copy(),
        label=y_test.copy(),
        cat_features=cat_features,
        datetime_features=date_features,
        index_name='id_column'
    )
    
    return train_dataset, test_dataset


def run_individual_checks(
    train_dataset: Dataset, 
    test_dataset: Dataset, 
    model: RandomForestClassifier
) -> Dict[str, Any]:
    """
    Run individual Deepchecks checks and return the results.
    
    Args:
        train_dataset: Deepchecks Dataset for training data
        test_dataset: Deepchecks Dataset for test data
        model: Trained model to validate
        
    Returns:
        Dictionary of check results
    """
    # Initialize results dictionary
    results = {}
    
    # Drift checks
    print("Running feature drift check...")
    feature_drift_check = TrainTestFeatureDrift()
    results['feature_drift'] = feature_drift_check.run(train_dataset, test_dataset)
    
    print("Running label drift check...")
    label_drift_check = TrainTestLabelDrift()
    results['label_drift'] = label_drift_check.run(train_dataset, test_dataset)
    
    print("Running multivariate drift check...")
    multivariate_drift_check = MultivariateDrift()
    results['multivariate_drift'] = multivariate_drift_check.run(train_dataset, test_dataset)
    
    # Data integrity checks
    print("Running duplicates check...")
    duplicates_check = DataDuplicates()
    results['data_duplicates'] = duplicates_check.run(train_dataset, test_dataset)
    
    print("Running mixed nulls check...")
    mixed_nulls_check = MixedNulls()
    results['mixed_nulls'] = mixed_nulls_check.run(train_dataset, test_dataset)
    
    print("Running class imbalance check...")
    class_imbalance_check = ClassImbalance()
    results['class_imbalance'] = class_imbalance_check.run(train_dataset)
    
    print("Running identifier leakage check...")
    identifier_leakage_check = IdentifierLeakage()
    results['identifier_leakage'] = identifier_leakage_check.run(train_dataset)
    
    # Model performance checks
    print("Running feature importance check...")
    # We need to handle categorical variables for the model
    X_train_model = train_dataset.data.drop(columns=['feature_3', 'date_col', 'id_column'])
    X_test_model = test_dataset.data.drop(columns=['feature_3', 'date_col', 'id_column'])
    feature_importance_check = FeatureImportance()
    results['feature_importance'] = feature_importance_check.run(train_dataset, test_dataset, model)
    
    print("Running model inference time check...")
    inference_time_check = ModelInferenceTime()
    results['inference_time'] = inference_time_check.run(train_dataset, model)
    
    # Classification specific checks
    print("Running confusion matrix report...")
    confusion_matrix_check = ConfusionMatrixReport()
    results['confusion_matrix'] = confusion_matrix_check.run(train_dataset, test_dataset, model)
    
    print("Running classification systematic error check...")
    systematic_error_check = ClassificationSystematicError()
    results['systematic_error'] = systematic_error_check.run(train_dataset, test_dataset, model)
    
    return results


def run_suites(
    train_dataset: Dataset, 
    test_dataset: Dataset, 
    model: RandomForestClassifier
) -> Dict[str, Any]:
    """
    Run Deepchecks pre-defined suites and return results.
    
    Args:
        train_dataset: Deepchecks Dataset for training data
        test_dataset: Deepchecks Dataset for test data
        model: Trained model to validate
        
    Returns:
        Dictionary of suite results
    """
    # Initialize results dictionary
    suite_results = {}
    
    # Model evaluation suite
    print("Running model evaluation suite...")
    X_train_model = train_dataset.data.drop(columns=['feature_3', 'date_col', 'id_column'])
    X_test_model = test_dataset.data.drop(columns=['feature_3', 'date_col', 'id_column'])
    model_eval_suite = model_evaluation()
    suite_results['model_evaluation'] = model_eval_suite.run(train_dataset, test_dataset, model)
    
    # Train-test validation suite
    print("Running train-test validation suite...")
    train_test_suite = train_test_validation()
    suite_results['train_test_validation'] = train_test_suite.run(train_dataset, test_dataset)
    
    # Data integrity suite
    print("Running data integrity suite...")
    integrity_suite = data_integrity()
    suite_results['data_integrity'] = integrity_suite.run(train_dataset)
    
    return suite_results


def save_validation_results(results: Dict[str, Any], suite_results: Dict[str, Any], output_dir: str = "./validation_results"):
    """
    Save validation results to files and log to MLflow.
    
    Args:
        results: Results from individual checks
        suite_results: Results from validation suites
        output_dir: Directory to save reports
    """
    import os
    import json
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save individual check results
    for check_name, check_result in results.items():
        # Save HTML report
        html_path = os.path.join(output_dir, f"{check_name}.html")
        check_result.save_as_html(html_path)
        
        # Log to MLflow
        with mlflow.start_run(run_name="deepchecks_validation"):
            # Log metrics if available
            if hasattr(check_result, 'value') and check_result.value is not None:
                try:
                    if isinstance(check_result.value, dict):
                        for k, v in check_result.value.items():
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                mlflow.log_metric(f"{check_name}_{k}", v)
                    elif isinstance(check_result.value, (int, float)) and not np.isnan(check_result.value):
                        mlflow.log_metric(check_name, check_result.value)
                except Exception as e:
                    print(f"Error logging metric for {check_name}: {e}")
            
            # Log artifacts
            mlflow.log_artifact(html_path)
    
    # Save suite results
    for suite_name, suite_result in suite_results.items():
        # Save HTML report
        html_path = os.path.join(output_dir, f"{suite_name}_suite.html")
        suite_result.save_as_html(html_path)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"deepchecks_{suite_name}_suite"):
            mlflow.log_artifact(html_path)


def model_validation_pipeline(config_path: str = None) -> Dict[str, bool]:
    """
    Run the complete model validation pipeline.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        Dictionary of validation results and pass/fail status
    """
    import yaml
    import os
    
    # Load configuration if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set thresholds from config or use defaults
    thresholds = config.get('thresholds', {
        'feature_drift_threshold': 0.05,
        'label_drift_threshold': 0.05,
        'data_duplicates_threshold': 0.01,
        'mixed_nulls_threshold': 0.05,
        'class_imbalance_threshold': 0.2,
    })
    
    try:
        # Set MLflow tracking URI if specified in config
        if 'mlflow_tracking_uri' in config:
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        
        # Load and prepare data
        print("Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Prepare datasets for Deepchecks
        print("Preparing Deepchecks datasets...")
        train_dataset, test_dataset = prepare_deepchecks_datasets(X_train, X_test, y_train, y_test)

        # Run individual checks
        print("Running individual checks...")
        individual_check_results = run_individual_checks(train_dataset, test_dataset, model)

        # Run pre-defined suites
        print("Running suites...")
        suite_results = run_suites(train_dataset, test_dataset, model)

        # Save the validation results and log to MLflow
        print("Saving validation results...")
        output_dir = config.get('output_dir', './validation_results')
        save_validation_results(individual_check_results, suite_results, output_dir)

        # Aggregate pass/fail status
        all_passed = True
        
        # Check individual checks against thresholds
        for check_name, check_result in individual_check_results.items():
            if not check_result.passed():
                print(f"Check failed: {check_name}")
                all_passed = False
        
        # Check suite results
        for suite_name, suite_result in suite_results.items():
            failed_checks = [check.check.__class__.__name__ 
                            for check in suite_result.get_check_results() 
                            if not check.passed()]
            if failed_checks:
                print(f"Suite {suite_name} has failed checks: {', '.join(failed_checks)}")
                all_passed = False

        print(f"Overall validation {'passed' if all_passed else 'failed'}")

        return {
            "individual_check_results": individual_check_results,
            "suite_results": suite_results,
            "overall_passed": all_passed
        }
        
    except Exception as e:
        print(f"Error in validation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {
            "individual_check_results": {},
            "suite_results": {},
            "overall_passed": False,
            "error": str(e)
        }


def main():
    """
    Main entry point for running the model validation pipeline from the command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Deepchecks Model Validation")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to optional configuration YAML file"
    )
    args = parser.parse_args()

    results = model_validation_pipeline(config_path=args.config)
    
    if results["overall_passed"]:
        print("Model validation succeeded!")
        exit(0)
    else:
        print("Model validation failed!")
        exit(1)


if __name__ == "__main__":
    main()

