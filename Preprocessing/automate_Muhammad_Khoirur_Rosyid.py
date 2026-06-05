import os
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer


# ==========================================
# CONFIGURATION
# ==========================================

RAW_DATA_PATH = "../Dataset_raw/ObesityDataSet.csv"

OUTPUT_DIR = "ObesityDataSet_preprocessing"

PROCESSED_DATA_PATH = os.path.join(
    OUTPUT_DIR,
    "ObesityDataSet_processed.csv"
)

LABEL_MAPPING_PATH = os.path.join(
    OUTPUT_DIR,
    "label_mapping.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

def load_data(path):

    print("=" * 50)
    print("LOADING DATASET")
    print("=" * 50)

    df = pd.read_csv(path)

    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================
# DATA CHECKING
# ==========================================

def check_data(df):

    print("\n" + "=" * 50)
    print("DATA CHECKING")
    print("=" * 50)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):

    print("\nRemoving duplicate rows...")

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Rows Before : {before}")
    print(f"Rows After  : {after}")
    print(f"Removed     : {before - after}")

    return df


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_data(df):

    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    # Feature dan Target
    X = df.drop(columns=["NObeyesdad"])

    y = df["NObeyesdad"]

    # --------------------------------------
    # Label Encoding Target
    # --------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    mapping_df = pd.DataFrame({
        "class_name": label_encoder.classes_,
        "label": range(len(label_encoder.classes_))
    })

    # --------------------------------------
    # Categorical & Numerical Columns
    # --------------------------------------

    categorical_features = (
        X.select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numerical_features = (
        X.select_dtypes(exclude=["object"])
        .columns
        .tolist()
    )

    print("\nCategorical Features:")
    print(categorical_features)

    print("\nNumerical Features:")
    print(numerical_features)

    # --------------------------------------
    # Transformer
    # --------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # --------------------------------------
    # Get Encoded Column Names
    # --------------------------------------

    encoded_columns = (
        preprocessor
        .named_transformers_["cat"]
        .get_feature_names_out(
            categorical_features
        )
    )

    final_columns = (
        numerical_features
        + encoded_columns.tolist()
    )

    processed_df = pd.DataFrame(
        X_processed,
        columns=final_columns
    )

    processed_df["NObeyesdad"] = y_encoded

    print(
        f"\nProcessed Dataset Shape: {processed_df.shape}"
    )

    return processed_df, mapping_df


# ==========================================
# SAVE OUTPUT
# ==========================================

def save_output(
    processed_df,
    mapping_df
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    processed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    mapping_df.to_csv(
        LABEL_MAPPING_PATH,
        index=False
    )

    print("\n" + "=" * 50)
    print("FILES SAVED")
    print("=" * 50)

    print(
        f"Processed Dataset : {PROCESSED_DATA_PATH}"
    )

    print(
        f"Label Mapping     : {LABEL_MAPPING_PATH}"
    )


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_data(
        RAW_DATA_PATH
    )

    check_data(df)

    df = remove_duplicates(df)

    processed_df, mapping_df = preprocess_data(df)

    save_output(
        processed_df,
        mapping_df
    )

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()