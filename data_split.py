from sklearn.model_selection import train_test_split
"""
    First split off test, then split remaining into train/val with adjusted val fraction.
"""
def train_val_test_split_data(X, y, val_size=0.2, test_size=0.2, seed=42):

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=seed, stratify=y_temp
    )
    print('Train set size: {}, Validation set size: {}, Test set size: {}'.format(
        len(X_train), len(X_val), len(X_test)
    ))
    return X_train, y_train, X_val, y_val, X_test, y_test