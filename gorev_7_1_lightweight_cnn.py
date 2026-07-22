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
EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE

BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "gorev_7_1_melspectrogram_raporu.xlsx"
OUTPUT_MODEL = BASE_DIR / "gorev_7_1_lightweight_cnn_model.keras"
OUTPUT_EXCEL = BASE_DIR / "gorev_7_1_sonuclar.xlsx"
OUTPUT_CM = BASE_DIR / "gorev_7_1_confusion_matrix.png"
OUTPUT_GRAPH = BASE_DIR / "gorev_7_1_training.png"

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


def print_excel_columns():
    excel_files = sorted(BASE_DIR.glob("*.xlsx"))
    print("[INFO] Klasördeki .xlsx dosyaları ve sütunları:")
    for path in excel_files:
        try:
            cols = pd.read_excel(path, nrows=0).columns.tolist()
        except Exception as exc:
            cols = [f"OKUMA HATASI: {exc}"]
        print("  ", path.name, cols)


def find_excel_file():
    if REPORT_FILE.exists():
        return REPORT_FILE

    excel_files = sorted(BASE_DIR.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"Klasörde hiç .xlsx dosyası yok: {BASE_DIR}")

    candidates = []
    for path in excel_files:
        try:
            df = pd.read_excel(path, nrows=0)
        except Exception:
            continue
        cols = [normalize(c) for c in df.columns]
        has_image = any(normalize(cand) in cols for cand in IMAGE_CANDIDATES)
        has_label = any(normalize(cand) in cols for cand in LABEL_CANDIDATES)
        if has_image and has_label:
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    print_excel_columns()
    if not candidates:
        raise FileNotFoundError(
            "Doğru Excel dosyası bulunamadı. Excel dosyanızda en az bir görsel sütunu ve bir etiket sütunu olmalı."
        )
    raise FileNotFoundError(
        "Birden fazla uygun Excel dosyası bulundu. Lütfen REPORT_FILE değişkenini doğru dosya adıyla ayarlayın.\n"
        f"Uygun dosyalar: {[p.name for p in candidates]}"
    )


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


REPORT_FILE = find_excel_file()
print(f"[INFO] Kullanılan Excel dosyası: {REPORT_FILE.name}")

df = pd.read_excel(REPORT_FILE)
image_column = find_column_name(df.columns, IMAGE_CANDIDATES)
label_column = find_column_name(df.columns, LABEL_CANDIDATES)

if image_column is None or label_column is None:
    print_excel_columns()
    raise ValueError(
        "Excel dosyasında gerekli sütunlar bulunamadı.\n"
        f"Bulunan sütunlar: {df.columns.tolist()}"
    )

df = df.dropna(subset=[image_column, label_column]).copy()
df["image_path"] = df[image_column].apply(resolve_image_path)
df["is_image"] = df["image_path"].apply(is_image_file)

invalid_paths = df.loc[~df["is_image"], "image_path"].unique().tolist()
if invalid_paths:
    print("[WARNING] Geçersiz veya bulunamayan resim yolları:")
    for p in invalid_paths[:20]:
        print("  ", p)

df = df[df["is_image"]].copy()
if df.empty:
    raise ValueError(
        "İşlenecek geçerli resim yolu kalmadı."
    )

counts = df[label_column].astype(str).value_counts()
rare_classes = counts[counts < 3].index.tolist()
if rare_classes:
    print(f"[WARNING] Çok az örneği olan sınıflar atılıyor: {rare_classes}")
    df = df[~df[label_column].astype(str).isin(rare_classes)].copy()

if df.empty:
    raise ValueError("Eğitim için yeterli veri yok.")

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df[label_column].astype(str))
classes = list(label_encoder.classes_)

if len(classes) < 2:
    raise ValueError("En az 2 farklı sınıf gereklidir.")

stratify_main = df["label_encoded"] if df["label_encoded"].value_counts().min() >= 2 else None
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=stratify_main,
    random_state=SEED,
)
stratify_val = train_df["label_encoded"] if train_df["label_encoded"].value_counts().min() >= 2 else None
train_df, val_df = train_test_split(
    train_df,
    test_size=0.20,
    stratify=stratify_val,
    random_state=SEED,
)

train_dataset = create_dataset(train_df["image_path"].values, train_df["label_encoded"].values, training=True)
validation_dataset = create_dataset(val_df["image_path"].values, val_df["label_encoded"].values)
test_dataset = create_dataset(test_df["image_path"].values, test_df["label_encoded"].values)

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
x = layers.Dropout(0.40)(x)

outputs = layers.Dense(len(classes), activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=train_df["label_encoded"].values,
)
class_weights = {i: float(w) for i, w in enumerate(weights)}

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(str(OUTPUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

predictions = model.predict(test_dataset, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = test_df["label_encoded"].to_numpy()

labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("=" * 60)
print("LIGHTWEIGHT CNN SONUÇLARI")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[classes[i] for i in labels],
    yticklabels=[classes[i] for i in labels],
)
plt.xlabel("Prediction")
plt.ylabel("True")
plt.title("Lightweight CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_GRAPH, dpi=300)
plt.close()

result_df = pd.DataFrame({
    "True_Label": label_encoder.inverse_transform(y_true),
    "Prediction": label_encoder.inverse_transform(y_pred),
})
result_df.to_excel(OUTPUT_EXCEL, index=False)

print()
print("=" * 60)
print("Classification Report")
print("=" * 60)
labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

print()
print("=" * 60)
print("Classification Report")
print("=" * 60)
print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=[classes[i] for i in labels],
    zero_division=0,
))