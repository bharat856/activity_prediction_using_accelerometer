from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
"""
    Proper KNN baseline:
      - Standardize flattened features.
      - Fit on train, predict on test.
"""
def _flat(X):
    return X.reshape(X.shape[0], -1)

def knn_classifier(X_train_seq, y_train, X_val_seq, y_val, X_test_seq, n_neighbors=5):
    X_train = _flat(X_train_seq)
    X_val   = _flat(X_val_seq)
    X_test  = _flat(X_test_seq)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    _ = scaler.transform(X_val)
    X_test_std  = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_std, y_train)
    y_pred_test = knn.predict(X_test_std)

    return {"model": knn, "y_pred_test": y_pred_test}