from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def is_performance_feature(column: str) -> bool:
    text = column.lower()
    return any(token in text for token in ("curricular units", "1st sem", "2nd sem", "1st semester", "2nd semester"))


def main():
    dataset = fetch_ucirepo(id=697)
    X_full = dataset.data.features.copy()
    y_frame = dataset.data.targets.copy()
    target_col = y_frame.columns[0]
    y = y_frame[target_col].copy()

    enrollment_cols = [c for c in X_full.columns if not is_performance_feature(c)]
    X = X_full[enrollment_cols].copy()

    categorical, numeric = [], []
    for column in X.columns:
        if X[column].dtype == "object":
            categorical.append(column)
        elif pd.api.types.is_integer_dtype(X[column]) and X[column].nunique(dropna=True) <= 50:
            categorical.append(column)
        else:
            numeric.append(column)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        [("cat", encoder, categorical), ("num", StandardScaler(), numeric)],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=SEED, stratify=y_train
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_e = label_encoder.fit_transform(y_train)
    y_val_e = label_encoder.transform(y_val)
    y_test_e = label_encoder.transform(y_test)

    classes = np.unique(y_train_e)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_e)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_p.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    model.fit(
        X_train_p,
        y_train_e,
        validation_data=(X_val_p, y_val_e),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(X_test_p, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test_e, predictions)
    macro_f1 = f1_score(y_test_e, predictions, average="macro")

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test_e, predictions, target_names=label_encoder.classes_))

    model.save(MODEL_DIR / "student_success_model.keras")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    metadata = {
        "target_col": target_col,
        "label_classes": list(label_encoder.classes_),
        "enrollment_features": enrollment_cols,
        "categorical_cols": categorical,
        "numeric_cols": numeric,
        "model_file": "models/student_success_model.keras",
        "preprocessor_file": "models/preprocessor.joblib",
        "label_encoder_file": "models/label_encoder.joblib",
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
