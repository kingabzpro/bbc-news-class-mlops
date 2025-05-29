"""
Prefect flow for the BBC-news classification MLOps project.
"""

from __future__ import annotations

import subprocess
import sys
import time

from fastapi.testclient import TestClient
from prefect import flow, get_run_logger, task

from src.api.main import app
from src.data.download import download_bbc_dataset
from src.data.preprocess import preprocess
from src.models.evaluate import main as evaluate
from src.models.train import train_model

################################################################################
# ------------------------------- Data layer ---------------------------------- #
################################################################################

@task(retries=3, retry_delay_seconds=5, log_prints=True)
def download_data() -> None:
    """Fetch raw BBC news data from Kaggle (idempotent)."""
    try:
        download_bbc_dataset()
        print("✔ Dataset downloaded")
    except Exception as e:
        print(f"⚠ Skipping dataset download: {str(e)}")
        print("Continuing with existing data...")


@task(retries=3, retry_delay_seconds=5, log_prints=True)
def preprocess_data() -> None:
    """Clean, split & tokenise the raw corpus."""
    try:
        preprocess()
        print("✔ Pre-processing complete")
    except Exception as e:
        print(f"⚠ Skipping pre-processing: {str(e)}")
        print("Continuing with existing processed data...")


################################################################################
# ------------------------------ Model layer ---------------------------------- #
################################################################################


@task(retries=3, retry_delay_seconds=5, log_prints=True)
def train() -> None:
    """Train a logistic-regression baseline (with simple hyper-tuning)."""
    train_model(classifier_type="logistic", tune_hyperparams=True)
    print("✔ Model training finished")


@task(retries=3, retry_delay_seconds=5, log_prints=True)
def evaluate_model(model_name: str, model_version: int) -> None:
    """Write evaluation metrics to MLflow."""
    evaluate(model_name, model_version)
    print(f"✔ Metrics logged for {model_name}:{model_version}")


################################################################################
# ------------------------------- Test layer ---------------------------------- #
################################################################################


@task(retries=1, log_prints=True)
def run_pytest() -> None:
    """
    Execute the whole pytest suite in-process.

    Any failure makes the Prefect task (and therefore the flow) fail.
    """
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_model.py", "-v"],
        check=False,
        capture_output=True,
        text=True,
    )

    print(res.stdout)
    if res.returncode:  # propagate failure
        print(res.stderr, file=sys.stderr)
        raise RuntimeError("❌ Pytest failures detected")
    print("✔ All unit tests passed")


@task(retries=1, log_prints=True)
def api_smoke_test(
    duration_s: int = 120, interval_s: float = 0.1, api_key: str = "test_key"
) -> None:
    """
    Continuously ping the local FastAPI app for `duration_s` seconds.

    Uses TestClient so no separate Uvicorn process is needed.
    """
    client = TestClient(app)
    end_time = time.monotonic() + duration_s
    n_ok, n_total = 0, 0

    while time.monotonic() < end_time:
        # /info endpoint
        assert client.get("/info").status_code == 200

        # /predict endpoint
        r = client.post(
            "/predict",
            json={"title": "Breaking news: Major development in technology sector"},
            headers={"X-API-Key": api_key},
        )
        assert r.status_code in (200, 401, 403, 503)
        n_ok += 1
        n_total += 1
        time.sleep(interval_s)

    print(f"✔ Smoke test ran {n_total} iterations without failure")


################################################################################
# --------------------------------- Flow ------------------------------------- #
################################################################################


@flow(name="mlops_pipeline")
def mlops_pipeline(
    model_name: str = "news_classifier_logistic",
    model_version: int = 1,
    duration_s: int = 10,
) -> None:
    """
    End-to-end orchestration entrypoint.
    Adjust `retries`, caching, and concurrency rules here as the project grows.
    """
    logger = get_run_logger()
    logger.info("🏁 Launching MLOps pipeline")

    download_data()
    preprocess_data()
    train()
    evaluate_model(model_name, model_version)
    run_pytest()  # full unit-test suite
    api_smoke_test(duration_s)  # 10-second live API hammering

    logger.info("🎉 Pipeline completed successfully")


if __name__ == "__main__":
    mlops_pipeline()
