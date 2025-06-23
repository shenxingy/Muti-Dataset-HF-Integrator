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
    """Check whether a folder contains a valid Huggingface dataset."""
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
        if "tensor" in key.lower() or "array" in key.lower():
            value = example[key]
            try:
                if isinstance(value, (list, tuple)) and len(value) > 100:
                    example[key] = f"<large {key} with {len(value)} elements>"
                elif hasattr(value, "__len__") and len(value) > 100:
                    example[key] = f"<large {key} with {len(value)} elements>"
            except Exception:
                example[key] = "<non-readable tensor-like field>"

    return example

def analyze_dataset(folder_path):
    """Load and analyze dataset schema, label values, and a sample example."""
    try:
        dataset = load_from_disk(folder_path)
        if isinstance(dataset, DatasetDict):
            dataset = dataset.get("train", next(iter(dataset.values())))

        features = dataset.features
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

        sample = dataset.shuffle(seed=42).select(range(min(50, len(dataset))))
        label_values = set(ex["label"] for ex in sample if "label" in ex)

        label_meanings = {}
        if "label" in features and features["label"].__class__.__name__ == "ClassLabel":
            label_meanings = {i: features["label"].int2str(i) for i in label_values}

        example = sanitize_large_fields(dataset[0])
        num_samples = len(dataset)

        return {
            "num_samples": num_samples,
            "features": feature_info,
            "label_unique_values": list(label_values),
            "label_meanings": label_meanings,
            "example": example
            
        }

    except Exception as e:
        return {"error": str(e)}

def default_serializer(obj):
    """Handle non-serializable objects during JSON export."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int_, np.intc, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
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
        print(f"[+] Processing: {name}")
        summary[name] = analyze_dataset(path)

    output_path = "local_dataset_preview.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"\n✅ Summary written to: {output_path}")

if __name__ == "__main__":
    main()
