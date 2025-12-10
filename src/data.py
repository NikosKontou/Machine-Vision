import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from . import config


def balance_dataset(df):
    """
    Upsamples minority classes to match the count of the majority class.
    This ensures the model sees an equal number of examples for each lesion type.
    """
    print("Balancing dataset (Upsampling)...")

    # 1. Find the maximum class count
    max_count = df['target'].value_counts().max()

    balanced_dfs = []

    # 2. Resample each class
    for class_name in df['target'].unique():
        class_subset = df[df['target'] == class_name]

        # Upsample (replace=True means we duplicate rows)
        df_resampled = resample(
            class_subset,
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        balanced_dfs.append(df_resampled)

    # 3. Combine back together
    df_balanced = pd.concat(balanced_dfs)

    # Shuffle the dataset so classes aren't grouped together
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Original size: {len(df)} -> Balanced size: {len(df_balanced)}")
    return df_balanced


def load_metadata(limit=None, balance=True):
    """
    Loads CSV, parses classes, prepares file paths, and optionally balances data.
    """
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

    # Apply limit first (if testing)
    if limit:
        print(f"Subsampling to {limit}...")
        actual_limit = min(limit, len(df))
        df, _ = train_test_split(df, train_size=actual_limit, stratify=df['label_idx'], random_state=42)

    # Apply balancing (Upsampling)
    if balance:
        df = balance_dataset(df)

    return df, available