import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    log_loss
)


# ==========================================
# DAGSHUB & MLFLOW CONFIGURATION
# ==========================================

os.environ["MLFLOW_TRACKING_USERNAME"] = "muhammadrosid77"
os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    "da2970cf9b321338c0f70f48816627787dde5122"
)

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/muhammadrosid77/"
    "obesity_clasification.mlflow"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Obesity_Classification_Baseline"

mlflow.set_experiment(EXPERIMENT_NAME)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("=" * 50)
    print("LOADING DATASET")
    print("=" * 50)

    df = pd.read_csv("ObesityDataSet_processed.csv")

    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]

    print(f"Dataset Shape  : {df.shape}")
    print(f"Features       : {X.shape[1]}")
    print(f"Target Classes : {y.nunique()}")

    return X, y


# ==========================================
# SPLIT DATA
# ==========================================

def split_data(X, y):

    print("\n" + "=" * 50)
    print("SPLITTING DATA")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train Size : {X_train.shape[0]}")
    print(f"Test Size  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test, y_pred
        ),
        "f1_weighted": f1_score(
            y_test, y_pred, average="weighted"
        ),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted"
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted"
        ),
        "cohen_kappa": cohen_kappa_score(
            y_test, y_pred
        ),
        "log_loss": log_loss(
            y_test, y_pred_proba
        )
    }

    report = classification_report(
        y_test, y_pred
    )

    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm, y_pred


# ==========================================
# TRAIN & LOG MODEL
# ==========================================

def train_and_log(
    model,
    model_name,
    params,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 50)
    print(f"TRAINING: {model_name}")
    print("=" * 50)

    with mlflow.start_run(
        run_name=f"Baseline_{model_name}"
    ):

        # ----------------------------------
        # Log Parameters
        # ----------------------------------

        mlflow.log_param("model_name", model_name)

        for key, value in params.items():
            mlflow.log_param(key, value)

        # ----------------------------------
        # Train Model
        # ----------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------
        # Evaluate
        # ----------------------------------

        metrics, report, cm, y_pred = evaluate_model(
            model, X_test, y_test
        )

        print(f"\nAccuracy           : {metrics['accuracy']:.4f}")
        print(f"F1 (weighted)      : {metrics['f1_weighted']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted)  : {metrics['recall_weighted']:.4f}")
        print(f"Cohen Kappa        : {metrics['cohen_kappa']:.4f}")
        print(f"Log Loss           : {metrics['log_loss']:.4f}")

        print(f"\nClassification Report:\n{report}")

        # ----------------------------------
        # Log Metrics
        # ----------------------------------

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # ----------------------------------
        # Artifact 1: Classification Report
        # ----------------------------------

        report_path = f"classification_report_{model_name}.txt"

        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # ----------------------------------
        # Artifact 2: Confusion Matrix
        # ----------------------------------

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()),
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"Confusion Matrix - {model_name}"
        )

        cm_path = f"confusion_matrix_{model_name}.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # ----------------------------------
        # Artifact 3: Feature Importance
        # ----------------------------------

        if hasattr(model, "feature_importances_"):

            importances = model.feature_importances_
            feature_names = X_train.columns

            feat_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(
                "importance", ascending=False
            ).head(15)

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(
                data=feat_imp,
                x="importance",
                y="feature",
                hue="feature",
                palette="viridis",
                legend=False,
                ax=ax
            )

            ax.set_title(
                f"Top 15 Feature Importance - "
                f"{model_name}"
            )
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")

            fi_path = (
                f"feature_importance_{model_name}.png"
            )
            fig.savefig(
                fi_path, dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            mlflow.log_artifact(fi_path)
            os.remove(fi_path)

        # ----------------------------------
        # Log Model
        # ----------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path=f"model_{model_name}"
        )

        print(f"\n[SUCCESS] {model_name} logged to MLflow!")

    return model, metrics


# ==========================================
# MAIN
# ==========================================

def main():

    # Load & Split
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    results = {}

    # ------------------------------------------
    # Model 1: Random Forest
    # ------------------------------------------

    rf_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }

    rf_model = RandomForestClassifier(
        **rf_params
    )

    rf_model, rf_metrics = train_and_log(
        model=rf_model,
        model_name="RandomForest",
        params=rf_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    results["RandomForest"] = rf_metrics

    # ------------------------------------------
    # Model 2: XGBoost
    # ------------------------------------------

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "use_label_encoder": False
    }

    xgb_model = XGBClassifier(
        **xgb_params
    )

    xgb_model, xgb_metrics = train_and_log(
        model=xgb_model,
        model_name="XGBoost",
        params=xgb_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    results["XGBoost"] = xgb_metrics

    # ------------------------------------------
    # Summary
    # ------------------------------------------

    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)

    for name, metrics in results.items():
        print(
            f"\n{name}:"
            f"\n  Accuracy    : {metrics['accuracy']:.4f}"
            f"\n  F1          : {metrics['f1_weighted']:.4f}"
            f"\n  Precision   : {metrics['precision_weighted']:.4f}"
            f"\n  Recall      : {metrics['recall_weighted']:.4f}"
            f"\n  Cohen Kappa : {metrics['cohen_kappa']:.4f}"
            f"\n  Log Loss    : {metrics['log_loss']:.4f}"
        )

    print("\n" + "=" * 50)
    print("BASELINE MODELLING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
