import os
from datasets import load_from_disk, concatenate_datasets, Features, Sequence, Value

# === CONFIGURATION ===
DATASET_ROOT = "/mnt/harddisk/voice_dataset"
SELECTED_DATASETS = [
    "CodecFake2",
    "VoxCelebSpoof_EdgeTTS",
    "VoxCelebSpoof_dataset",
    "XTTS_data",
    "asvspoof_la_10256",
    "openvoice_2000_v2_varlen",
    "for_dataset_final",
    "mlaad_20000",
]
OUTPUT_PATH = os.path.join(DATASET_ROOT, "merged_spoof_dataset")

# === TARGET FEATURE SCHEMA ===
TARGET_FEATURES = Features({
    "audio_tensor": Sequence(Sequence(Value("float32"))),
    "sample_rate": Value("int64"),
    "label": Value("string"),
    "attack_type": Value("string"),
    "attack_model": Value("string"),
    "duration": Value("float64"),
    "file_path": Value("string"),
    "utt_id": Value("string"),
    "prompt": Value("string"),
    "original_file": Value("string"),
    "transcript": Value("string"),
    "source": Value("string"),
})

# === FUNCTIONS ===
def load_and_tag_dataset(dataset_name):
    dataset_path = os.path.join(DATASET_ROOT, dataset_name)
    print(f"📦 Loading: {dataset_name}")
    dataset = load_from_disk(dataset_path)

    if isinstance(dataset, dict):
        dataset = dataset.get("train", next(iter(dataset.values())))

    # Filter only spoof/bonafide
    dataset = dataset.filter(lambda x: x.get("label") in {"spoof", "bonafide"}, desc=f"Filtering {dataset_name}")

    # Cast to aligned features
    dataset = dataset.cast(TARGET_FEATURES, allow_missing=True)

    # Add source field
    dataset = dataset.map(lambda x: {"source": dataset_name}, desc=f"Tagging source: {dataset_name}")

    return dataset

def main():
    print("🔄 Loading and merging datasets...")
    datasets = [load_and_tag_dataset(name) for name in SELECTED_DATASETS]
    merged = concatenate_datasets(datasets)

    print(f"\n✅ Merged dataset: {len(merged)} samples.")
    print(f"🧾 Final Features: {list(merged.features.keys())}")

    print(f"💾 Saving to: {OUTPUT_PATH}")
    merged.save_to_disk(OUTPUT_PATH)
    print("✅ Done.")

if __name__ == "__main__":
    main()
