import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pickle
import boto3


print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


HISTORY_PATH = "outputs/best_history.pkl"
CSV_PATH= "architecture/data/data.csv"
OUTPUT_DIR= "outputs"
MODEL_PATH= os.path.join(OUTPUT_DIR, "best_model.keras")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
N_TRIALS = 150
TEST_SIZE = 0.20
RANDOM_STATE= 42

S3_BUCKET= "aws-ml-foundations"
S3_PREFIX= "proyecto/splits/"
AWS_REGION= "us-east-2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


CATEGORICAL_COLS = ["category", "urgency", "region"]
DROP_COLS = ["contract_id"]
TARGET_COL= "delayed"


def preprocess(df: pd.DataFrame):
    df = df.drop(columns=DROP_COLS)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=float)

    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    feature_names = df.drop(columns=[TARGET_COL]).columns.tolist()
    print(f"  Features after encoding : {X.shape[1]}")
    print(f"  Feature names           : {feature_names}\n")
    return X, y, feature_names


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    return X_train, X_test, y_train, y_test, scaler


def upload_splits_to_s3(X_train, X_test, y_train, y_test, feature_names):


    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    splits = {
        "X_train.csv": pd.DataFrame(X_train, columns=feature_names),
         "X_test.csv":  pd.DataFrame(X_test,  columns=feature_names),
         "y_train.csv": pd.DataFrame(y_train, columns=["delayed"]),
         "y_test.csv":  pd.DataFrame(y_test,  columns=["delayed"]),
     }
    
    for filename, data in splits.items():
         local_path = os.path.join(OUTPUT_DIR, filename)
         data.to_csv(local_path, index=False)
         s3.upload_file(local_path, S3_BUCKET, S3_PREFIX + filename)
         print(f"  [S3] Uploaded {filename}")
    
    s3.upload_file(SCALER_PATH, S3_BUCKET, S3_PREFIX + "scaler.pkl")


def build_model(trial, n_features: int) -> keras.Model:
    n_layers      = trial.suggest_int("n_layers", 3, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))

    for i in range(n_layers):
        units   = trial.suggest_categorical(f"units_l{i}", [16,32, 64, 128])
        dropout = trial.suggest_float(f"dropout_l{i}", 0.4, 0.6)
        model.add(layers.Dense(units, 
                            activation="relu",
                            kernel_regularizer=keras.regularizers.l2(trial.suggest_float("l2", 1e-5, 1e-2, log=True))))
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_objective(X_train, y_train, n_features):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs     = trial.suggest_int("epochs", 20, 80)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=trial.number,
        )

        model = build_model(trial, n_features)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        y_pred_prob = model.predict(X_val, verbose=0).ravel()
        return roc_auc_score(y_val, y_pred_prob)

    return objective


def train_best_model(best_params, X_train, y_train, n_features):
    fixed_trial = optuna.trial.FixedTrial(best_params)
    model = build_model(fixed_trial, n_features)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            previous_history = pickle.load(f)
        initial_epoch = len(previous_history["loss"])
        print(f"  Resuming final training from epoch {initial_epoch}")
        model = keras.models.load_model(MODEL_PATH)
    else:
        initial_epoch = 0

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        initial_epoch=initial_epoch,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            previous_history = pickle.load(f)
        for key in history.history:
            history.history[key] = previous_history[key] + history.history[key]

    model.save(MODEL_PATH)

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    return model, history


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def evaluate(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    spec = specificity_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["On time", "Delayed"]))

    return {
        "accuracy": acc, "precision": precision,
        "recall": recall, "specificity": spec,
        "roc_auc": auc, "y_prob": y_prob, "y_pred": y_pred,
    }


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0).ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="Neural Net")
    ax.set_title("ROC Curve — IT Procurement Delay Prediction")
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    labels = ["On time", "Delayed"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)

    upload_splits_to_s3(X_train, X_test, y_train, y_test, feature_names)

    study = optuna.create_study(
        study_name="it_procurement_v4",
        storage="sqlite:///outputs/optuna_study.db",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        load_if_exists=True,
    )
    print(f"  Resuming from trial {len(study.trials)}" if len(study.trials) > 0 else "  Starting fresh")

    study.optimize(
        make_objective(X_train, y_train, X_train.shape[1]),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    model, history = train_best_model(study.best_params, X_train, y_train, X_train.shape[1])

    results = evaluate(model, X_test, y_test,threshold=0.4)

    plot_training_history(history)
    plot_roc_curve(model, X_test, y_test)
    plot_confusion_matrix(y_test, results["y_pred"])


if __name__ == "__main__":
    main()