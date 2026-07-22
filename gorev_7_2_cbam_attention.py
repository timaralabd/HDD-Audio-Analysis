import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "gorev_7_1_melspectrogram_raporu.xlsx"

OUTPUT_MODEL_BASELINE = BASE_DIR / "gorev_7_2_baseline_model.keras"
OUTPUT_MODEL_CBAM = BASE_DIR / "gorev_7_2_cbam_model.keras"
OUTPUT_CM_BASELINE = BASE_DIR / "gorev_7_2_baseline_cm.png"
OUTPUT_CM_CBAM = BASE_DIR / "gorev_7_2_cbam_cm.png"
OUTPUT_GRAPH = BASE_DIR / "gorev_7_2_training.png"

IMAGE_CANDIDATES = [
    "melgörüntüsü", "melgoruntusu", "mel_goruntusu", "mel_spektrogram",
    "mel_spectrogram", "image", "imagepath", "path", "görüntü", "goruntu",
    "file", "filename"
]
LABEL_CANDIDATES = [
    "hatasınıfı", "hatasinifi", "hata_sinifi", "label",
    "class", "sinif", "sınıf", "target"
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def find_column_name(columns, candidates):
    normalized = {normalize(c): c for c in columns}
    for cand in candidates:
        key = normalize(cand)
        if key in normalized:
            return normalized[key]
    return None


def resolve_image_path(value):
    path_str = str(value).strip()
    if not path_str:
        return ""
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = BASE_DIR / path_str
    if candidate.exists():
        return str(candidate)
    found = next((p for p in BASE_DIR.rglob(path.name) if p.is_file()), None)
    return str(found) if found else ""


def is_image_file(path):
    path = Path(path)
    return path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


def create_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=max(100, len(paths)), seed=SEED)
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def cbam_block(x, ratio=8):
    channel = x.shape[-1]
    shared = layers.Dense(max(1, channel // ratio), activation="relu")
    shared_out = layers.Dense(channel)

    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared(avg_pool)
    avg_pool = shared_out(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared(max_pool)
    max_pool = shared_out(max_pool)

    channel_att = layers.Add()([avg_pool, max_pool])
    channel_att = layers.Activation("sigmoid")(channel_att)
    x = layers.Multiply()([x, channel_att])

    avg_pool_sp = layers.Lambda(lambda z: tf.reduce_mean(z, axis=3, keepdims=True))(x)
    max_pool_sp = layers.Lambda(lambda z: tf.reduce_max(z, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_pool_sp, max_pool_sp])
    spatial_att = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(concat)
    x = layers.Multiply()([x, spatial_att])
    return x


def build_baseline(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


def build_cbam(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = cbam_block(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


def train_and_evaluate(model, name, train_ds, val_ds, test_ds, test_df, classes, class_weights, output_model, output_cm):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(str(output_model), monitor="val_accuracy", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    y_true = test_df["label_encoded"].to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[classes[i] for i in labels],
        yticklabels=[classes[i] for i in labels],
    )
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_cm, dpi=300)
    plt.close()

    print()
    print("=" * 60)
    print(f"{name} SONUÇLARI")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[classes[i] for i in labels],
        zero_division=0,
    ))
    return history


def main():
    if not REPORT_FILE.exists():
        raise FileNotFoundError(f"Excel dosyası yok: {REPORT_FILE}")

    df = pd.read_excel(REPORT_FILE)
    image_column = find_column_name(df.columns, IMAGE_CANDIDATES)
    label_column = find_column_name(df.columns, LABEL_CANDIDATES)

    if image_column is None or label_column is None:
        raise ValueError(f"Excel sütunları bulunamadı: {df.columns.tolist()}")

    df = df.dropna(subset=[image_column, label_column]).copy()
    df["image_path"] = df[image_column].apply(resolve_image_path)
    df["is_image"] = df["image_path"].apply(is_image_file)

    invalid = df.loc[~df["is_image"], "image_path"].unique().tolist()
    if invalid:
        print("[WARNING] Geçersiz resim yolları (ilk 20):")
        for p in invalid[:20]:
            print(" ", p)

    df = df[df["is_image"]].copy()
    if df.empty:
        raise ValueError("Geçerli resim yok.")

    counts = df[label_column].astype(str).value_counts()
    rare_classes = counts[counts < 2].index.tolist()
    if rare_classes:
        print(f"[WARNING] Atılan nadir sınıflar: {rare_classes}")
        df = df[~df[label_column].astype(str).isin(rare_classes)].copy()

    if df.empty:
        raise ValueError("Eğitim için yeterli veri yok.")

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df[label_column].astype(str))
    classes = list(label_encoder.classes_)

    counts = df["label_encoded"].value_counts()
    stratify_main = df["label_encoded"] if counts.min() >= 2 else None

    try:
        train_df, test_df_local = train_test_split(
            df,
            test_size=0.20,
            stratify=stratify_main,
            random_state=SEED,
        )
    except ValueError:
        print("[WARNING] Stratified split failed — stratify=None ile devam ediliyor.")
        train_df, test_df_local = train_test_split(
            df,
            test_size=0.20,
            stratify=None,
            random_state=SEED,
        )

    stratify_val = train_df["label_encoded"] if train_df["label_encoded"].value_counts().min() >= 2 else None
    try:
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.20,
            stratify=stratify_val,
            random_state=SEED,
        )
    except ValueError:
        print("[WARNING] Stratified val split failed — stratify=None ile devam ediliyor.")
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.20,
            stratify=None,
            random_state=SEED,
        )

    train_ds = create_dataset(train_df["image_path"].values, train_df["label_encoded"].values, training=True)
    val_ds = create_dataset(val_df["image_path"].values, val_df["label_encoded"].values)
    test_ds = create_dataset(test_df_local["image_path"].values, test_df_local["label_encoded"].values)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=train_df["label_encoded"].values,
    )
    class_weights = {i: float(w) for i, w in enumerate(weights)}

    baseline = build_baseline(len(classes))
    cbam = build_cbam(len(classes))

    train_and_evaluate(
        baseline,
        "Baseline CNN",
        train_ds,
        val_ds,
        test_ds,
        test_df_local,
        classes,
        class_weights,
        OUTPUT_MODEL_BASELINE,
        OUTPUT_CM_BASELINE,
    )

    train_and_evaluate(
        cbam,
        "CBAM CNN",
        train_ds,
        val_ds,
        test_ds,
        test_df_local,
        classes,
        class_weights,
        OUTPUT_MODEL_CBAM,
        OUTPUT_CM_CBAM,
    )

    print("\nConfusion matrisleri kaydedildi:")
    print("Baseline:", OUTPUT_CM_BASELINE.name)
    print("CBAM    :", OUTPUT_CM_CBAM.name)


if __name__ == "__main__":
    main()