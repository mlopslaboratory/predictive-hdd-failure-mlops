import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score


def train_model(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    return model, roc, f1


def run_training(X_train, y_train, X_test, y_test):

    mlflow.set_experiment("hdd_failure")

    with mlflow.start_run():

        model, roc, f1 = train_model(X_train, y_train, X_test, y_test)

        # logging params
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 200)

        # logging metrics
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("f1", f1)

        # logging model
        mlflow.sklearn.log_model(model, "model")

    return model