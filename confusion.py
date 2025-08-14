from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Display a confusion matrix heatmap.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Title of the plot
        label_names: Optional list of class names for axis ticks
"""
def display_confusion_matrix(y_true, y_pred, title, label_names=None):
    cm = confusion_matrix(y_true, y_pred, labels=label_names if label_names is not None else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names if label_names is not None else True,
                yticklabels=label_names if label_names is not None else True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

"""
    Confusion matrix for CNN integer outputs with string class names.
"""
def display_confusion_matrix_cnn(y_true_int, y_pred_int, label_names):
    cm = confusion_matrix(y_true_int, y_pred_int, labels=range(len(label_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


"""
    Display a separate confusion matrix for each class vs. all others.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label values
        label_names: Optional list of class names
"""
def display_confusion_matrices_per_class(y_true, y_pred, labels, label_names=None):
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_true == label, y_pred == label)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not ' + (label_names[i] if label_names else str(label)),
                                 (label_names[i] if label_names else str(label))],
                    yticklabels=['Not ' + (label_names[i] if label_names else str(label)),
                                 (label_names[i] if label_names else str(label))])
        plt.title(f'Class: {label_names[i] if label_names else str(label)}')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()