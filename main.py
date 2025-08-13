import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

class SeqDataset(Dataset):
    def __init__(self, X_seq, y_int):
        self.X = torch.from_numpy(X_seq.transpose(0, 2, 1)).float()
        self.y = torch.from_numpy(y_int).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.classifier(x)

def train_cnn(X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test,
              num_classes, epochs=8, batch_size=128, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SeqDataset(X_train_seq, y_train)
    val_ds   = SeqDataset(X_val_seq, y_val)
    test_ds  = SeqDataset(X_test_seq, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = CNN1D(in_channels=3, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_model = None
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                pred_labels = torch.argmax(preds, dim=1)
                correct += (pred_labels == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    if best_model:
        model.load_state_dict(best_model)

    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            pred_labels = torch.argmax(preds, dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_true.extend(yb.numpy())

    return model, np.array(all_preds), np.array(all_true)

def load_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'activity', 'x-axis', 'y-axis', 'z-axis'])
    df = df.sort_values(['user', 'activity', 'timestamp'])
    print(f'Data set loaded with {len(df)} records.')
    return df

def train_val_test_split_data(X, y, val_size=0.2, test_size=0.2, seed=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=seed)
    print(f'Train set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_windows(df, window_size=100, step_size=64, max_windows_per_class=None):
    X_list, y_list = [], []
    for (user, activity), group in df.groupby(['user', 'activity']):
        data = group[['x-axis', 'y-axis', 'z-axis']].to_numpy(dtype=np.float32)
        n_samples = len(data)
        count = 0
        for start in range(0, n_samples - window_size + 1, step_size):
            if max_windows_per_class and count >= max_windows_per_class:
                break
            window = data[start:start+window_size]
            X_list.append(window)
            y_list.append(activity)
            count += 1
    X_seq = np.array(X_list)
    y = np.array(y_list)
    return X_seq, y

def random_forest_classifier(X_train_seq, y_train, X_val_seq, y_val, X_test_seq, n_estimator=300, max_depth=None, seed=42):
    def flat(X): return X.reshape(X.shape[0], -1)
    X_train = flat(X_train_seq)
    X_val = flat(X_val_seq)
    X_test = flat(X_test_seq)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=seed, n_jobs=-1)
    rf.fit(X_train_std, y_train)

    y_pred_test = rf.predict(X_test_std)
    return {
        "model": rf,
        "y_pred_test": y_pred_test
    }

def knn_classifier(X_train_seq, y_train, X_val_seq, y_val, X_test_seq, n_neighbors=5):
    def flat(X): return X.reshape(X.shape[0], -1)
    X_train = flat(X_train_seq)
    X_val = flat(X_val_seq)
    X_test = flat(X_test_seq)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    t, c = 128, 3
    X_train_seq_std = X_train_std.reshape(-1, t, c)
    X_val_seq_std = X_val_std.reshape(-1, t, c)
    X_test_seq_std = X_test_std.reshape(-1, t, c)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_std, y_train)

    return {
        "X_train_seq_std": X_train_seq_std,
        "X_val_seq_std": X_val_seq_std,
        "X_test_seq_std": X_test_seq_std
    }

def cnn_classifier(knn_out, y_train, y_val, y_test, epochs=8, batch_size=64, lr=0.001):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    model, y_pred, y_true = train_cnn(knn_out["X_train_seq_std"], y_train_enc,
                                      knn_out["X_val_seq_std"], y_val_enc,
                                      knn_out["X_test_seq_std"], y_test_enc,
                                      num_classes=len(le.classes_),
                                      epochs=epochs, batch_size=batch_size, lr=lr)
    return {
        "model": model,
        "preds": y_pred,
        "true": y_true,
        "encoder": le
    }

def compute_metric(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def display_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == "__main__":
    path = "Data/time_series_data_human_activities.csv"
    df = load_data(path)
    X_seq, y = create_windows(df, window_size=128, step_size=64, max_windows_per_class=800)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_data(X_seq, y)

    rf_out = random_forest_classifier(X_train, y_train, X_val, y_val, X_test)
    print("Random Forest Metrics:")
    rf_metrics = compute_metric(y_test, rf_out["y_pred_test"])
    print(rf_metrics)
    display_confusion_matrix(y_test, rf_out["y_pred_test"], title="Random Forest Confusion Matrix")
    print("\nRandom Forest - Predicted vs Actual:")
    for i in range(10):
        print(f"Pred: {rf_out['y_pred_test'][i]}  |  Actual: {y_test[i]}")

    knn_out = knn_classifier(X_train, y_train, X_val, y_val, X_test)
    cnn_out = cnn_classifier(knn_out, y_train, y_val, y_test)
    print("\nCNN Metrics:")
    cnn_metrics = compute_metric(cnn_out["true"], cnn_out["preds"])
    print(cnn_metrics)
    display_confusion_matrix(cnn_out["true"], cnn_out["preds"], title="CNN Confusion Matrix")
    print("\nCNN - Predicted vs Actual:")
    for i in range(10):
        print(f"Pred: {cnn_out['preds'][i]}  |  Actual: {cnn_out['true'][i]}")