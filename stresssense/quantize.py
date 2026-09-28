import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR


def representative_dataset(X_train, sample_count=2000):
    indices = np.linspace(
        0,
        len(X_train) - 1,
        min(sample_count, len(X_train)),
        dtype=int,
    )
    for index in indices:
        yield [X_train[index : index + 1].astype(np.float32)]


def convert(model, X_train, full_int8):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X_train)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    if full_int8:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    return converter.convert()


def evaluate(model_content, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    predictions = []

    for sample in X_test:
        value = sample[None].astype(np.float32)
        if input_info["dtype"] == np.int8:
            scale, zero_point = input_info["quantization"]
            value = np.clip(
                np.round(value / scale + zero_point), -128, 127
            ).astype(np.int8)

        interpreter.set_tensor(input_info["index"], value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])

        if output_info["dtype"] == np.int8:
            scale, zero_point = output_info["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        predictions.append(int(np.argmax(output)))

    return float(np.mean(np.asarray(predictions) == y_test))


def main():
    model_path = MODELS_DIR / "stresssense_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\nRun train.py first."
        )

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    model = keras.models.load_model(model_path)
    _, float_accuracy = model.evaluate(X_test, y_test, verbose=0)

    full_int8 = convert(model, X_train, full_int8=True)
    hybrid = convert(model, X_train, full_int8=False)
    full_path = MODELS_DIR / "stresssense_full_int8.tflite"
    hybrid_path = MODELS_DIR / "stresssense_hybrid.tflite"
    full_path.write_bytes(full_int8)
    hybrid_path.write_bytes(hybrid)

    metrics = {
        "float_accuracy": float(float_accuracy),
        "full_int8_accuracy": evaluate(full_int8, X_test, y_test),
        "hybrid_accuracy": evaluate(hybrid, X_test, y_test),
        "full_int8_size_bytes": len(full_int8),
        "hybrid_size_bytes": len(hybrid),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "quantization_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
