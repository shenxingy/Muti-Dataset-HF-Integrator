import os
from datasets import load_from_disk, concatenate_datasets

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

# === FUNCTIONS ===
def load_and_tag_dataset(dataset_name):
    dataset_path = os.path.join(DATASET_ROOT, dataset_name)
    dataset = load_from_disk(dataset_path)

    # Support DatasetDict (e.g., train/test split)
    if isinstance(dataset, dict):
        dataset = dataset.get("train", next(iter(dataset.values())))

    # Add source tag
    dataset = dataset.map(lambda x: {"source": dataset_name}, desc=f"Tagging source: {dataset_name}")

    # Only keep spoof or bonafide
    dataset = dataset.filter(lambda x: x.get("label") in {"spoof", "bonafide"}, desc="Filtering valid labels")

    return dataset

def main():
    print("🔄 Loading and merging datasets...")
    datasets = [load_and_tag_dataset(name) for name in SELECTED_DATASETS]
    merged = concatenate_datasets(datasets)

    print(f"✅ Merged dataset: {len(merged)} samples.")
    print(f"🧾 Features: {list(merged.features.keys())}")

    print(f"💾 Saving to: {OUTPUT_PATH}")
    merged.save_to_disk(OUTPUT_PATH)
    print("✅ Done.")

if __name__ == "__main__":
    main()
