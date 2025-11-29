import numpy as np
from src import data, features, model


def main():
    # 1. Load Data
    df, classes = data.load_metadata(limit=None)  # Set limit=None for full run

    # 2. Feature Extraction Loop
    print(f"Extracting features for {len(df)} images...")
    X = []
    y = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 100 == 0: print(f"Processing {idx}/{total}...")

        # Call the pipeline from the package
        feats = features.extract_all_features_pipeline(row['path'])

        if feats is not None:
            X.append(feats)
            y.append(row['label_idx'])

    # 3. Train & Save
    if len(X) > 0:
        model.train_and_evaluate(np.array(X), np.array(y), classes)
    else:
        print("No features extracted. Check image paths.")


if __name__ == "__main__":
    main()