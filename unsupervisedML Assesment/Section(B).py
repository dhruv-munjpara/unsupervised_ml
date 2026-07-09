# ===============================
# IBM HR Attrition Data Loader
# ===============================

import pandas as pd


def load_dataset(file_path):
    """
    Load CSV dataset
    """
    df = pd.read_csv(file_path)
    return df


def dataset_summary(df):
    """
    Display basic dataset information
    """
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())


def split_features_label(df, target="Attrition"):
    """
    Split Features (X) and Label (y)
    """
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def main():

    file_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    df = load_dataset(file_path)

    dataset_summary(df)

    X, y = split_features_label(df)

    print("\nFeature Shape:", X.shape)
    print("Label Shape:", y.shape)

    print("\nFirst Five Features")
    print(X.head())

    print("\nFirst Five Labels")
    print(y.head())


if __name__ == "__main__":
    main()