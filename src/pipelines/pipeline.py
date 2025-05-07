"""
Prefect pipeline for orchestrating the MLOps workflow.
"""

import subprocess
import sys

from prefect import flow, task


@task
def download_data():
    subprocess.run([sys.executable, "-m", "src.data.download"], check=True)


@task
def preprocess_data():
    subprocess.run([sys.executable, "-m", "src.data.preprocess"], check=True)


@task
def train_model():
    subprocess.run([sys.executable, "-m", "src.models.train"], check=True)


@task
def evaluate_model():
    subprocess.run([sys.executable, "-m", "src.models.evaluate"], check=True)


@flow
def mlops_pipeline():
    download_data()
    preprocess_data()
    train_model()
    evaluate_model()


if __name__ == "__main__":
    mlops_pipeline()
