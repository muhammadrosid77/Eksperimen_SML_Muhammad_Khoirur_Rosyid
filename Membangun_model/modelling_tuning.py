import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

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

EXPERIMENT_NAME = "Obesity_Classification_Tuning"

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
# HYPERPARAMETER TUNING
# ==========================================

def tune_model(
    model_class,
    model_name,
    param_distributions,
    X_train,
    X_test,
    y_train,
    y_test,
    n_iter=20
):

    print("\n" + "=" * 50)
    print(f"TUNING: {model_name}")
    print("=" * 50)

    # ------------------------------------------
    # RandomizedSearchCV
    # ------------------------------------------

    search = RandomizedSearchCV(
        estimator=model_class,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    # ------------------------------------------
    # Log Setiap Kombinasi ke MLflow
    # ------------------------------------------

    print(f"\nLogging {len(search.cv_results_['params'])} "
          f"combinations to MLflow...")

    mlflow.autolog()

    cv_results = search.cv_results_

    for i in range(len(cv_results["params"])):

        with mlflow.start_run(
            run_name=f"Tuning_{model_name}_iter_{i+1}"
        ):

            mlflow.log_param(
                "model_name", model_name
            )

            mlflow.log_param(
                "tuning_iteration", i + 1
            )

            # Log Parameters
            for key, value in cv_results["params"][i].items():
                mlflow.log_param(key, value)

            # Log CV Metrics
            mlflow.log_metric(
                "mean_cv_accuracy",
                cv_results["mean_test_score"][i]
            )

            mlflow.log_metric(
                "std_cv_accuracy",
                cv_results["std_test_score"][i]
            )

            mlflow.log_metric(
                "cv_rank",
                int(cv_results["rank_test_score"][i])
            )

    # ------------------------------------------
    # Log Best Model
    # ------------------------------------------

    print(f"\nBest Parameters: {search.best_params_}")
    print(f"Best CV Score  : {search.best_score_:.4f}")

    best_model = search.best_estimator_

    metrics, report, cm, y_pred = evaluate_model(
        best_model, X_test, y_test
    )

    print(f"\nTest Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted)      : {metrics['f1_weighted']:.4f}")
    print(f"Test Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Test Recall (weighted)  : {metrics['recall_weighted']:.4f}")
    print(f"Test Cohen Kappa        : {metrics['cohen_kappa']:.4f}")
    print(f"Test Log Loss           : {metrics['log_loss']:.4f}")
    print(f"\nClassification Report:\n{report}")

    with mlflow.start_run(
        run_name=f"Best_{model_name}_Tuned"
    ):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("best_cv_score", search.best_score_)

        for key, value in search.best_params_.items():
            mlflow.log_param(f"best_{key}", value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Artifact 1: Classification Report
        report_path = (
            f"classification_report_best_{model_name}.txt"
        )

        with open(report_path, "w") as f:
            f.write(
                f"Best Parameters:\n"
                f"{search.best_params_}\n\n"
                f"Best CV Score: "
                f"{search.best_score_:.4f}\n\n"
                f"Classification Report:\n"
                f"{report}"
            )

        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # Artifact 2: Confusion Matrix
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
            f"Confusion Matrix - Best {model_name}"
        )

        cm_path = (
            f"confusion_matrix_best_{model_name}.png"
        )
        fig.savefig(
            cm_path, dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Artifact 3: Feature Importance
        if hasattr(best_model, "feature_importances_"):

            importances = best_model.feature_importances_
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
                f"Best {model_name}"
            )
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")

            fi_path = (
                f"feature_importance_best_"
                f"{model_name}.png"
            )
            fig.savefig(
                fi_path, dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            mlflow.log_artifact(fi_path)
            os.remove(fi_path)

        # Log Model
        mlflow.sklearn.log_model(
            best_model,
            artifact_path=f"best_model_{model_name}"
        )

        print(f"\n[SUCCESS] Best {model_name} "
              f"logged to MLflow!")

    return best_model, metrics, search.best_params_


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
    # Tuning 1: Random Forest
    # ------------------------------------------

    rf_param_distributions = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    rf_model, rf_metrics, rf_best_params = tune_model(
        model_class=RandomForestClassifier(
            random_state=42
        ),
        model_name="RandomForest",
        param_distributions=rf_param_distributions,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_iter=20
    )

    results["RandomForest"] = {
        "metrics": rf_metrics,
        "best_params": rf_best_params
    }

    # ------------------------------------------
    # Tuning 2: XGBoost
    # ------------------------------------------

    xgb_param_distributions = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3, 0.5]
    }

    xgb_model, xgb_metrics, xgb_best_params = tune_model(
        model_class=XGBClassifier(
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False
        ),
        model_name="XGBoost",
        param_distributions=xgb_param_distributions,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_iter=20
    )

    results["XGBoost"] = {
        "metrics": xgb_metrics,
        "best_params": xgb_best_params
    }

    # ------------------------------------------
    # Summary
    # ------------------------------------------

    print("\n" + "=" * 50)
    print("TUNING RESULTS SUMMARY")
    print("=" * 50)

    for name, data in results.items():

        metrics = data["metrics"]
        best_params = data["best_params"]

        print(
            f"\n{name}:"
            f"\n  Best Params : {best_params}"
            f"\n  Accuracy    : {metrics['accuracy']:.4f}"
            f"\n  F1          : {metrics['f1_weighted']:.4f}"
            f"\n  Precision   : {metrics['precision_weighted']:.4f}"
            f"\n  Recall      : {metrics['recall_weighted']:.4f}"
            f"\n  Cohen Kappa : {metrics['cohen_kappa']:.4f}"
            f"\n  Log Loss    : {metrics['log_loss']:.4f}"
        )

    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
