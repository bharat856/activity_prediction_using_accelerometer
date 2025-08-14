import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

"""
    Standardizes flattened features, trains RF on train set, predicts on test.
"""

def _flat(X):
    return X.reshape(X.shape[0], -1)

def random_forest_classifier(X_train_seq, y_train, X_val_seq, y_val, X_test_seq, n_estimators=300, max_depth=None, seed=42):
    X_train = _flat(X_train_seq)
    X_val   = _flat(X_val_seq)
    X_test  = _flat(X_test_seq)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    _ = scaler.transform(X_val)
    X_test_std  = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=seed, n_jobs=-1)
    rf.fit(X_train_std, y_train)
    y_pred_test = rf.predict(X_test_std)

    return {"model": rf, "y_pred_test": y_pred_test}