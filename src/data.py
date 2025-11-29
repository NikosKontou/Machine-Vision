import pandas as pd
import os
from sklearn.model_selection import train_test_split
from . import config


def load_metadata(limit=None):
    """Loads CSV, parses classes, and prepares file paths."""
    print("Loading Metadata...")
    if not os.path.exists(config.CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {config.CSV_PATH}")

    df = pd.read_csv(config.CSV_PATH)

    # Validate Classes
    classes = config.CLASSES
    available = [c for c in classes if c in df.columns]

    df['target'] = df[available].idxmax(axis=1)
    df['label_idx'] = df['target'].apply(lambda x: available.index(x))
    df['path'] = df['image'].apply(lambda x: os.path.join(config.IMAGE_FOLDER, x + '.jpg'))

    if limit:
        print(f"Subsampling to {limit}...")
        actual_limit = min(limit, len(df))
        df, _ = train_test_split(df, train_size=actual_limit, stratify=df['label_idx'], random_state=42)

    return df, available