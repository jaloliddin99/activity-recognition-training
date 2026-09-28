import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp/matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

from config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-copies", type=int, default=1)
    return parser.parse_args()


def augment(X, rng, noise_std=0.05, max_shift=10):
    result = X.copy()
    result += rng.normal(0, noise_std, result.shape).astype(np.float32)
    scales = rng.uniform(0.95, 1.05, (len(result), 1, 1))
    result *= scales.astype(np.float32)
    for index in range(len(result)):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        result[index] = np.roll(result[index], shift, axis=0)
    return result


def build_model(input_shape, num_classes):
    l2 = regularizers.l2(1e-4)
    return models.Sequential([
        layers.Input(shape=input_shape, name="input"),
        layers.Conv1D(
            32, 5, padding="same", activation="relu",
            kernel_regularizer=l2, name="conv1",
        ),
        layers.MaxPooling1D(2, name="pool1"),
        layers.Dropout(0.15, name="drop1"),
        layers.Conv1D(
            64, 5, padding="same", activation="relu",
            kernel_regularizer=l2, name="conv2",
        ),
        layers.MaxPooling1D(2, name="pool2"),
        layers.Dropout(0.2, name="drop2"),
        layers.Conv1D(
            64, 3, padding="same", activation="relu",
            kernel_regularizer=l2, name="conv3",
        ),
        layers.MaxPooling1D(2, name="pool3"),
        layers.GlobalAveragePooling1D(name="global_average_pool"),
        layers.Dense(64, activation="relu", kernel_regularizer=l2, name="dense"),
        layers.Dropout(0.4, name="drop3"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])


def plot_results(history, y_true, y_pred, class_names):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "training_history.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix(y_true, y_pred)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, values, title, fmt in [
        (axes[0], matrix, "Confusion Matrix", "d"),
        (axes[1], normalized, "Normalized Confusion Matrix", ".2f"),
    ]:
        sns.heatmap(
            values,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_validation = np.load(PROCESSED_DIR / "X_validation.npy")
    y_validation = np.load(PROCESSED_DIR / "y_validation.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    with open(PROCESSED_DIR / "preprocessing_info.pkl", "rb") as handle:
        preprocessing_info = pickle.load(handle)

    label_to_activity = preprocessing_info["label_to_activity"]
    class_names = [label_to_activity[i] for i in range(len(label_to_activity))]

    training_parts = [X_train]
    label_parts = [y_train]
    for _ in range(args.augmentation_copies):
        training_parts.append(augment(X_train, rng))
        label_parts.append(y_train)
    X_augmented = np.concatenate(training_parts).astype(np.float32)
    y_augmented = np.concatenate(label_parts)
    order = rng.permutation(len(X_augmented))
    X_augmented = X_augmented[order]
    y_augmented = y_augmented[order]

    model = build_model(X_train.shape[1:], len(class_names))
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        X_augmented,
        y_augmented,
        validation_data=(X_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report_text = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "stresssense_model.h5")
    plot_results(history, y_test, predictions, class_names)

    training_info = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "epochs_trained": len(history.history["accuracy"]),
        "total_params": int(model.count_params()),
        "input_shape": tuple(X_train.shape[1:]),
        "num_classes": len(class_names),
        "label_to_activity": label_to_activity,
        "config": preprocessing_info["config"],
    }
    with open(MODELS_DIR / "training_info.pkl", "wb") as handle:
        pickle.dump(training_info, handle)
    with open(RESULTS_DIR / "classification_report.txt", "w") as handle:
        handle.write(report_text)
    with open(RESULTS_DIR / "metrics.json", "w") as handle:
        json.dump(
            {**training_info, "classification_report": report_dict},
            handle,
            indent=2,
            default=list,
        )

    print("\n" + report_text)
    print(f"Cross-user float accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
