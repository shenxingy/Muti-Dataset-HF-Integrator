import os
from datasets import load_from_disk, concatenate_datasets, Features, Sequence, Value, Dataset
from tqdm.auto import tqdm # For better progress bars

# === CONFIGURATION ===
DATASET_ROOT = "/home/ubuntu/scam-ai-0/voice_dataset"
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
# Keep this as is, it's well-defined.
TARGET_FEATURES = Features({
    "audio_tensor": Sequence(Sequence(Value("float32"))),
    "sample_rate": Value("int64"),
    "label": Value("string"),
    "attack_type": Value("string"),
    "attack_model": Value("string"),
    "duration": Value("float64"),
    "source": Value("string"),
})

def get_default_value_for_type(feature_type):
    if isinstance(feature_type, Value):
        if feature_type.dtype.startswith("int"):
            return 0
        elif feature_type.dtype.startswith("float"):
            return 0.0
        elif feature_type.dtype == "string":
            return ""
    elif isinstance(feature_type, Sequence):
        return []
    else:
        return None

# === FUNCTIONS ===
def load_and_process_dataset(dataset_name):
    dataset_path = os.path.join(DATASET_ROOT, dataset_name)
    print(f"📦 Loading: {dataset_name} from {dataset_path}")

    # load_from_disk might return DatasetDict if it has splits (e.g., 'train', 'validation')
    loaded_data = load_from_disk(dataset_path)

    # Ensure we get a single Dataset object.
    # Prioritize 'train' split if available, otherwise take the first available split,
    # or if it's already a Dataset, use it directly.
    if isinstance(loaded_data, dict):
        if "train" in loaded_data:
            dataset = loaded_data["train"]
        else:
            # Fallback to the first available split if 'train' is not found
            dataset = next(iter(loaded_data.values()))
    else:
        dataset = loaded_data # It's already a Dataset object

    print(f"🔄 Processing {dataset_name} (samples: {len(dataset)})")

    # 1. Align schema: add missing columns with default values, remove extra columns.
    existing_cols = set(dataset.column_names)
    target_cols = set(TARGET_FEATURES.keys())

    missing_cols = list(target_cols - existing_cols)
    cols_to_remove = list(existing_cols - target_cols)

    # Remove extra columns
    if cols_to_remove:
        print(f"   - Removing {len(cols_to_remove)} columns: {cols_to_remove}")
        dataset = dataset.remove_columns(cols_to_remove)

    # Add missing columns (except 'source', which is added next)
    if 'source' in missing_cols:
        missing_cols.remove('source') # 'source' is special, added with dataset_name

    if missing_cols:
        print(f"   - Adding {len(missing_cols)} missing columns: {missing_cols}")
        def fill_missing(batch):
            for col in missing_cols:
                default_value = get_default_value_for_type(TARGET_FEATURES[col])
                batch[col] = [default_value] * len(next(iter(batch.values())))
            return batch
        dataset = dataset.map(fill_missing, batched=True, desc=f"Filling missing columns for {dataset_name}")

    # 2. Add source field using a batched map for efficiency.
    dataset = dataset.map(
        lambda batch: {"source": [dataset_name] * len(batch["audio_tensor"])},
        batched=True,
        desc=f"Tagging source: {dataset_name}"
    )


    # 3. Cast features to ensure type consistency before concatenating
    dataset = dataset.cast(TARGET_FEATURES)

    return dataset

def main():
    print("🔄 Loading and merging datasets...")
    # Use tqdm for better progress visualization during the loading loop
    loaded_datasets = []
    for name in tqdm(SELECTED_DATASETS, desc="Overall Dataset Loading"):
        try:
            ds = load_and_process_dataset(name)
            loaded_datasets.append(ds)
        except Exception as e:
            print(f"⚠️ Error loading or processing {name}: {e}. Skipping this dataset.")

    if not loaded_datasets:
        print("❌ No datasets were loaded successfully. Exiting.")
        return

    print("\n🔗 Concatenating datasets...")
    merged = concatenate_datasets(loaded_datasets)

    print(f"\n✅ Merged dataset: {len(merged)} samples.")
    print(f"🧾 Final Features: {list(merged.features.keys())}")

    print(f"💾 Saving to: {OUTPUT_PATH}")
    # Optimization 3: Use num_proc for saving to disk if you have multiple CPU cores
    # This can significantly speed up writing large datasets to disk.
    # Adjust `num_proc` based on your available CPU cores.
    merged.save_to_disk(OUTPUT_PATH, num_proc=os.cpu_count() or 4) # Use all available cores, or default to 4

    print("✅ Done.")

if __name__ == "__main__":
    main()