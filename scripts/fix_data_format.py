#!/usr/bin/env python3
"""Flatten nested input fields in training data for LLaMA Factory compatibility."""

import json
from pathlib import Path


def flatten_input(input_field):
    """Convert nested input dict to a formatted string."""
    if isinstance(input_field, str):
        return input_field
    if not isinstance(input_field, dict):
        return str(input_field)

    parts = []
    if "contexts" in input_field:
        contexts = input_field["contexts"]
        labels = input_field.get("labels", [])
        for i, ctx in enumerate(contexts):
            label = labels[i] if i < len(labels) else ""
            if label:
                parts.append(f"[{label}] {ctx}")
            else:
                parts.append(ctx)

    return "\n\n".join(parts) if parts else ""


def process_file(input_path, output_path):
    """Process a single JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    for item in data:
        if "input" in item:
            item["input"] = flatten_input(item["input"])

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {input_path} -> {output_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    for filename in ["medical_train.json", "medical_val.json", "medical_test.json"]:
        input_path = data_dir / filename
        if input_path.exists():
            # Backup original
            backup_path = data_dir / f"{filename}.bak"
            if not backup_path.exists():
                import shutil

                shutil.copy(input_path, backup_path)
            process_file(input_path, input_path)
