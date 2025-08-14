import numpy as np
"""
    Slices each (user, activity) series into overlapping windows.
    Returns:
      X_seq: (N, T, 3) float32
      y:     (N,) label strings
"""
def create_windows(df, window_size=100, step_size=64, max_windows_per_class=None):
    X_list, y_list = [], []
    for (user, activity), group in df.groupby(['user', 'activity']):
        data = group[['x-axis', 'y-axis', 'z-axis']].to_numpy(dtype=np.float32)
        n_samples = len(data)
        count = 0
        for start in range(0, n_samples - window_size + 1, step_size):
            if max_windows_per_class and count >= max_windows_per_class:
                break
            window = data[start:start + window_size]  # (T, 3)
            X_list.append(window)
            y_list.append(activity)
            count += 1

    X_seq = np.array(X_list)
    y = np.array(y_list)
    return X_seq, y