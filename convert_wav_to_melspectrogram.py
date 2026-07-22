import re
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "melspectrogram_images"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_CANDIDATES = [
    "melgörüntüsü", "melgoruntusu", "mel_goruntusu", "mel_spektrogram",
    "mel_spectrogram", "image", "imagepath", "path", "görüntü", "goruntu",
    "file", "filename"
]
LABEL_CANDIDATES = [
    "hatasınıfı", "hatasinifi", "hata_sinifi", "label",
    "class", "sinif", "sınıf", "target"
]


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def find_excel_file():
    excels = sorted(BASE_DIR.glob("*.xlsx"))
    if not excels:
        raise FileNotFoundError("Bu klasörde .xlsx dosyası yok.")
    for path in excels:
        try:
            df = pd.read_excel(path, nrows=0)
        except Exception:
            continue
        cols = [normalize(c) for c in df.columns]
        has_image = any(normalize(cand) in cols for cand in IMAGE_CANDIDATES)
        has_label = any(normalize(cand) in cols for cand in LABEL_CANDIDATES)
        if has_image and has_label:
            return path
    raise FileNotFoundError(
        "Uygun Excel dosyası bulunamadı. Excel sütunları File/Path ve Class/Label içermeli."
    )


def find_column_name(columns, candidates):
    normalized = {normalize(c): c for c in columns}
    for cand in candidates:
        key = normalize(cand)
        if key in normalized:
            return normalized[key]
    return None


def save_melspectrogram(wav_path, image_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis=None,
        y_axis=None,
        cmap="viridis",
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def find_wav_file(wav_value):
    wav_path = Path(wav_value)
    if not wav_path.is_absolute():
        wav_path = BASE_DIR / wav_path

    if wav_path.exists():
        return wav_path

    found = next(
        (p for p in BASE_DIR.rglob(wav_path.name) if p.is_file()), None
    )
    if found:
        return found

    raise FileNotFoundError(f"WAV dosyası bulunamadı: {wav_value}")


def main():
    excel_path = find_excel_file()
    print(f"[INFO] Kullanılan Excel dosyası: {excel_path.name}")

    df = pd.read_excel(excel_path)
    file_col = find_column_name(df.columns, IMAGE_CANDIDATES)
    label_col = find_column_name(df.columns, LABEL_CANDIDATES)

    if file_col is None or label_col is None:
        raise ValueError(
            f"Excel dosyasında gerekli sütunlar yok: {df.columns.tolist()}"
        )

    new_paths = []
    for _, row in df.iterrows():
        wav_value = str(row[file_col]).strip()
        if not wav_value or wav_value.lower() in {"nan", "none"}:
            new_paths.append("")
            continue

        wav_path = find_wav_file(wav_value)
        image_path = OUTPUT_DIR / f"{wav_path.stem}.png"
        save_melspectrogram(wav_path, image_path)
        new_paths.append(str(image_path.relative_to(BASE_DIR)))

    result_df = pd.DataFrame({
        "Mel Görüntüsü": new_paths,
        "Hata Sınıfı": df[label_col].astype(str),
    })

    output_excel = BASE_DIR / "gorev_7_1_melspectrogram_raporu.xlsx"
    result_df.to_excel(output_excel, index=False)
    print(f"[INFO] Mel-spektrogram Excel hazırlandı: {output_excel}")
    print(f"[INFO] Resimler kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()