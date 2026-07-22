import tensorflow as tf
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "gorev_7_1_lightweight_cnn_model.keras"
    tflite_path = base_dir / "gorev_7_1_lightweight_cnn_model.tflite"
    tflite_quant_path = base_dir / "gorev_7_1_lightweight_cnn_model_quant.tflite"

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant = converter.convert()
    tflite_quant_path.write_bytes(tflite_quant)

    print("TFLite model saved:", tflite_path.name)
    print("Quantized TFLite model saved:", tflite_quant_path.name)

if __name__ == "__main__":
    main()