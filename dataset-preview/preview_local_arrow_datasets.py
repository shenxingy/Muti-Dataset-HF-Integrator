import os
import json
import numpy as np
from datasets import load_from_disk, DatasetDict

# === Configuration ===
DATASET_ROOT = "/mnt/harddisk/voice_dataset"
os.environ["HF_DATASETS_OFFLINE"] = "1"  # Avoid network calls
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"

# === Utility Functions ===

def is_arrow_dataset(folder_path):
    """Check whether a folder contains a valid Hugging Face dataset (Arrow format)."""
    return (
        os.path.exists(os.path.join(folder_path, "dataset_info.json")) and
        any(f.endswith(".arrow") for f in os.listdir(folder_path))
    )

def sanitize_large_fields(example):
    """Remove or summarize large array-like fields to prevent bloated output."""
    if "audio" in example and isinstance(example["audio"], dict):
        example["audio"]["array"] = f"<{len(example['audio']['array'])} samples>"

    if "audio_tensor" in example:
        tensor = example["audio_tensor"]
        try:
            length = len(tensor)
            example["audio_tensor"] = f"<tensor with {length} elements>"
        except TypeError:
            example["audio_tensor"] = "<audio tensor removed>"

    for key in list(example.keys()):
        value = example[key]
        if "tensor" in key.lower() or "array" in key.lower():
            try:
                if isinstance(value, (list, tuple)) and len(value) > 100:
                    example[key] = f"<large {key} with {len(value)} elements>"
                elif hasattr(value, "__len__") and len(value) > 100:
                    example[key] = f"<large {key} with {len(value)} elements>"
            except Exception:
                example[key] = "<non-readable tensor-like field>"

    return example

def analyze_dataset(folder_path):
    """Load a Hugging Face dataset from disk and analyze its structure and contents."""
    try:
        dataset = load_from_disk(folder_path)

        # Analyze splits
        split_presence = {
            "train": False,
            "test": False,
            "validation": False
        }
        all_splits = []

        if isinstance(dataset, DatasetDict):
            all_splits = list(dataset.keys())
            for split in split_presence:
                split_presence[split] = split in dataset

            dataset_used = dataset.get("train", next(iter(dataset.values())))
        else:
            dataset_used = dataset
            split_presence["train"] = True
            all_splits = ["train"]

        # Extract feature info
        features = dataset_used.features
        feature_info = {}

        for key, feature in features.items():
            if feature.__class__.__name__ == "ClassLabel":
                feature_info[key] = {
                    "type": "ClassLabel",
                    "num_classes": feature.num_classes,
                    "names": feature.names
                }
            else:
                feature_info[key] = f"ValueType: {str(feature.dtype)}"

        # Sample analysis
        sample = dataset_used.shuffle(seed=42).select(range(min(50, len(dataset_used))))
        label_values = set(ex["label"] for ex in sample if "label" in ex)

        label_meanings = {}
        if "label" in features and features["label"].__class__.__name__ == "ClassLabel":
            label_meanings = {i: features["label"].int2str(i) for i in label_values}

        # Build return structure
        return {
            "num_samples": len(dataset_used),
            "features": feature_info,
            "label_unique_values": sorted(label_values),
            "label_meanings": label_meanings,
            "example": sanitize_large_fields(dataset_used[0]),
            "split_presence": split_presence,
            "all_splits": all_splits
        }

    except Exception as e:
        return {
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "folder": folder_path
            }
        }

def default_serializer(obj):
    """Handle non-serializable objects during JSON export."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    return str(obj)

# === Main Execution ===

def main():
    summary = {}

    for name in sorted(os.listdir(DATASET_ROOT)):
        path = os.path.join(DATASET_ROOT, name)
        if not os.path.isdir(path) or not is_arrow_dataset(path):
            continue
        print(f"[+] Processing dataset: {name}")
        summary[name] = analyze_dataset(path)

    output_path = "local_dataset_preview.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"\n✅ Summary written to: {output_path}")

if __name__ == "__main__":
    main()
