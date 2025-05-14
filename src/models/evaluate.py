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
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
model_path = next(MODEL_DIR.glob("news_classifier_*.joblib"), None)

if model_path is not None:
    model = joblib.load(model_path)
    X_val = val_df["title"]
    y_val = val_df["category"]
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_val, y_pred, output_dict=True)
    report_str = classification_report(y_val, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y_val))
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

    # Log Loss and ROC AUC
    roc_path = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_val)
        try:
            ll = log_loss(y_val, y_pred_proba, labels=model.classes_)
        except Exception as e:
            print(f"Could not calculate log loss: {e}")
        y_val_binarized = label_binarize(y_val, classes=model.classes_)
        n_classes = y_val_binarized.shape[1]
        fig_roc, ax_roc = plt.subplots()
        if n_classes > 1:
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
            try:
                roc_auc_macro = roc_auc_score(
                    y_val,
                    y_pred_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=model.classes_,
                )
            except Exception as e:
                print(f"Could not calculate macro ROC AUC: {e}")
        elif n_classes == 1:
            positive_class_proba = (
                y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
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
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC)")
        ax_roc.legend(loc="lower right")
        roc_path = MODEL_DIR / "roc_auc_curves.png"
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)
    else:
        print("Model does not support predict_proba. Skipping log loss and ROC AUC.")

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
