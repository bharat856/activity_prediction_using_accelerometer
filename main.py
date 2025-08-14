from data_loading import load_data
from creating_windows import create_windows
from data_split import train_val_test_split_data
from rf import random_forest_classifier
from knn import knn_classifier
from cnn_classifier import cnn_classifier
from metrics import compute_metrics, per_class_report
from confusion import display_confusion_matrix, display_confusion_matrices_per_class, display_confusion_matrix_cnn

def main():
    path = "Data/time_series_data_human_activities.csv"

    #Load & window
    df = load_data(path)
    X_seq, y = create_windows(df, window_size=128, step_size=64, max_windows_per_class=800)

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_data(X_seq, y)

 
    # Random Forest 
    rf_out = random_forest_classifier(X_train, y_train, X_val, y_val, X_test)
    print("\nRandom Forest Metrics:")
    rf_metrics = compute_metrics(y_test, rf_out["y_pred_test"])
    print(rf_metrics)

    labels_rf = sorted(list(set(y_test))) 
    display_confusion_matrix(y_test, rf_out["y_pred_test"],
                             title="Random Forest Confusion Matrix",
                             label_names=labels_rf)

    display_confusion_matrices_per_class(y_test, rf_out["y_pred_test"],
                                         labels=labels_rf, label_names=labels_rf)

    rf_report = per_class_report(y_test, rf_out["y_pred_test"], label_names=labels_rf)
    print("\nRandom Forest per-class report:")
    for cls_name, stats in rf_report.items():
        if cls_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(cls_name, stats)


    # KNN 

    knn_out = knn_classifier(X_train, y_train, X_val, y_val, X_test)
    print("\nKNN Metrics:")
    knn_metrics = compute_metrics(y_test, knn_out["y_pred_test"])
    print(knn_metrics)

    display_confusion_matrix(y_test, knn_out["y_pred_test"],
                             title="KNN Confusion Matrix",
                             label_names=labels_rf)

    display_confusion_matrices_per_class(y_test, knn_out["y_pred_test"],
                                         labels=labels_rf, label_names=labels_rf)

    knn_report = per_class_report(y_test, knn_out["y_pred_test"], label_names=labels_rf)
    print("\nKNN per-class report:")
    for cls_name, stats in knn_report.items():
        if cls_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(cls_name, stats)

    #CNN
    cnn_out = cnn_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                             epochs=8, batch_size=64, lr=0.001)

    print("\nCNN Metrics:")
    cnn_metrics = compute_metrics(cnn_out["true"], cnn_out["preds"])
    print(cnn_metrics)

    label_names_cnn = list(cnn_out["encoder"].classes_)  
    labels_cnn = list(range(len(label_names_cnn)))       

    display_confusion_matrix_cnn(cnn_out["true"], cnn_out["preds"], label_names_cnn)

    display_confusion_matrices_per_class(cnn_out["true"], cnn_out["preds"],
                                         labels=labels_cnn, label_names=label_names_cnn)

    cnn_report = per_class_report(cnn_out["true"], cnn_out["preds"], label_names=label_names_cnn)
    print("\nCNN per-class report:")
    for cls_name, stats in cnn_report.items():
        if cls_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(cls_name, stats)


if __name__ == "__main__":
    main()