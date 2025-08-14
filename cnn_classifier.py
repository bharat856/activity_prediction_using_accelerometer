import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from train_cnn import train_cnn


"""
      1) Label-encode string labels -> ints.
      2) Standardize features: flatten (N, T*C), fit StandardScaler on train,
         transform all, then reshape back to (N, T, C).
      3) Call train_cnn (which expects (N, T, C) and int labels).
"""
def cnn_classifier(X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, epochs=8, batch_size=64, lr=0.001):
    # 1) label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # 2) standardization + reshape back
    N, T, C = X_train_seq.shape
    flat = lambda X: X.reshape(X.shape[0], -1)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(flat(X_train_seq))
    X_val_std   = scaler.transform(flat(X_val_seq))
    X_test_std  = scaler.transform(flat(X_test_seq))

    X_train_seq_std = X_train_std.reshape(-1, T, C)
    X_val_seq_std   = X_val_std.reshape(-1, T, C)
    X_test_seq_std  = X_test_std.reshape(-1, T, C)

    # 3) train/eval
    model, y_pred_int, y_true_int = train_cnn(
        X_train_seq_std, y_train_enc,
        X_val_seq_std,   y_val_enc,
        X_test_seq_std,  y_test_enc,
        num_classes=len(le.classes_),
        epochs=epochs, batch_size=batch_size, lr=lr
    )

    return {
        "model": model,
        "preds": y_pred_int,
        "true": y_true_int,
        "encoder": le
    }