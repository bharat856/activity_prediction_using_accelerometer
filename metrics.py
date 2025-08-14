"""
- Overall accuracy 
- Precision/Recall/F1 
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
)

"""
    accuracy, weighted precision, and weighted F1 score.
"""
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

"""
    Per-class precision, accuracy, and F1 score.
    Args:
        label_names: Optional list of class names. If None, numeric labels are used.
"""
def per_class_report(y_true, y_pred, label_names=None):
    if label_names is None:
        label_names = sorted(list(set(y_true)))

    report = {}
    for label in label_names:
        mask = np.array(y_true) == label
        class_precision = precision_score(y_true, y_pred, labels=[label], average="micro", zero_division=0)
        class_f1 = f1_score(y_true, y_pred, labels=[label], average="micro", zero_division=0)
        class_accuracy = accuracy_score(mask, np.array(y_pred) == label)

        report[label] = {
            "accuracy": class_accuracy,
            "precision": class_precision,
            "f1_score": class_f1
        }

    return report