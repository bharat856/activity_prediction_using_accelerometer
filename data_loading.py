import pandas as pd
"""
        Reads the CSV, coerces timestamp, drops missing required cols,
        and sorts by (user, activity, timestamp) so time windows are ordered.
"""
def load_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'activity', 'x-axis', 'y-axis', 'z-axis'])
    df = df.sort_values(['user', 'activity', 'timestamp'])
    print(f'Data set loaded with {len(df)} records.')
    return df