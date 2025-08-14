import torch
import numpy as np
from torch.utils.data import DataLoader
from to_torch import SeqDataset
from cnn_definition import CNN1D
import torch.nn as nn

"""
    Pure training/eval loop.
    Assumes inputs are already standardized and shaped as (N, T, C).
"""

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

    model = CNN1D(in_channels=X_train_seq.shape[-1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred_labels = model(xb).argmax(dim=1)
                correct += (pred_labels == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    # test
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.numpy())

    return model, np.array(all_preds), np.array(all_true)