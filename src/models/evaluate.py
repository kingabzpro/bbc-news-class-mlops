"""
Evaluate the trained model using scikit-learn.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
model_path = next(MODEL_DIR.glob("news_classifier_*.joblib"), None)

if model_path is not None:
    model = joblib.load(model_path)
    X_val = val_df["title"]
    y_val = val_df["category"]
    # If your model expects vectorized input, you may need to load the vectorizer and transform X_val
    # For now, assuming model can handle raw text
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    report_str = classification_report(y_val, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_val),
        yticklabels=np.unique(y_val),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve (only for binary classification)
    roc_path = None
    if len(np.unique(y_val)) == 2:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_score, pos_label=np.unique(y_val)[1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            roc_path = MODEL_DIR / "roc_curve.png"
            plt.savefig(roc_path)
            plt.close()

    # Markdown Report
    md_path = MODEL_DIR / "model_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Accuracy:** {acc:.4f}\n\n")
        f.write("## Classification Report\n")
        f.write("```\n" + report_str + "\n```\n\n")
        f.write("## Confusion Matrix\n")
        f.write(f"![Confusion Matrix]({cm_path.name})\n\n")
        if roc_path:
            f.write("## ROC Curve\n")
            f.write(f"![ROC Curve]({roc_path.name})\n\n")
    print(f"Markdown evaluation report saved to {md_path}")
else:
    print("No trained model found for evaluation.")
