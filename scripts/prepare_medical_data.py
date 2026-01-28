"""
Medical dataset preparation script
Combines MedQA, PubMedQA, and MedDialog into unified format
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def clean_medical_text(text: str) -> str:
    """Clean medical text and remove PHI patterns"""
    # Remove potential PHI
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)  # SSN
    text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE]", text)  # Phone

    # Standardize medical terminology spacing
    text = text.replace("mg", " mg")
    text = text.replace("mL", " mL")

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def format_medqa(sample: dict) -> dict:
    """Format MedQA sample to Alpaca format"""
    question = sample["question"]
    options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample["options"])])
    answer_idx = sample["answer_idx"]
    answer = sample["options"][answer_idx]
    explanation = sample.get("explanation", "")

    output = (
        f"The correct answer is {chr(65+answer_idx)}. {answer}\n\n"
        f"{explanation}\n\n"
        "[Educational purposes only. Consult healthcare provider for medical advice.]"
    )

    return {
        "instruction": f"Answer the following medical question:\n\n{question}\n\n{options}",
        "input": "",
        "output": clean_medical_text(output),
    }


def format_pubmedqa(sample: dict) -> dict:
    """Format PubMedQA sample to Alpaca format"""
    output = (
        f"{sample['final_decision']}\n\n"
        f"{sample['long_answer']}\n\n"
        "[Based on published research. Consult healthcare provider.]"
    )

    return {
        "instruction": sample["question"],
        "input": sample.get("context", ""),
        "output": clean_medical_text(output),
    }


def format_meddialog(sample: dict) -> dict | None:
    """Format MedDialog sample to Alpaca format"""
    utterances = sample.get("utterances", [])
    if len(utterances) < 2:
        return None

    output = utterances[1] + "\n\n[Educational purposes only.]"

    return {
        "instruction": utterances[0],  # Patient question
        "input": "",
        "output": clean_medical_text(output),
    }


def main():
    """Main data preparation pipeline"""
    print("🏥 Medical LLM Data Preparation")
    print("=" * 50)

    # Create output directories
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📥 Loading datasets...")

    # Load datasets
    try:
        medqa = load_dataset("bigbio/med_qa", split="train")
        print(f"✓ Loaded MedQA: {len(medqa)} samples")
    except Exception as e:
        print(f"⚠ Could not load MedQA: {e}")
        medqa = []

    try:
        pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        print(f"✓ Loaded PubMedQA: {len(pubmedqa)} samples")
    except Exception as e:
        print(f"⚠ Could not load PubMedQA: {e}")
        pubmedqa = []

    try:
        meddialog = load_dataset("medical_dialog", "en", split="train")
        print(f"✓ Loaded MedDialog: {len(meddialog)} samples")
    except Exception as e:
        print(f"⚠ Could not load MedDialog: {e}")
        meddialog = []

    print("\n🔄 Processing and formatting...")

    # Combine datasets
    combined = []

    # Sample from MedQA (30% of target 50K = 15K)
    if medqa:
        samples = min(15000, len(medqa))
        combined.extend([format_medqa(s) for s in random.sample(list(medqa), samples)])
        print(f"✓ Processed {samples} MedQA samples")

    # Sample from PubMedQA (20% = 10K)
    if pubmedqa:
        samples = min(10000, len(pubmedqa))
        combined.extend([format_pubmedqa(s) for s in random.sample(list(pubmedqa), samples)])
        print(f"✓ Processed {samples} PubMedQA samples")

    # Sample from MedDialog (30% = 15K)
    if meddialog:
        samples = min(15000, len(meddialog))
        for s in random.sample(list(meddialog), samples):
            formatted = format_meddialog(s)
            if formatted:
                combined.append(formatted)
        print("✓ Processed MedDialog samples")

    # Shuffle
    random.shuffle(combined)

    print(f"\n📊 Total samples: {len(combined)}")

    # Split (80/10/10)
    train_split = int(0.8 * len(combined))
    val_split = int(0.9 * len(combined))

    train_data = combined[:train_split]
    val_data = combined[train_split:val_split]
    test_data = combined[val_split:]

    print(f"   Train: {len(train_data)}")
    print(f"   Val:   {len(val_data)}")
    print(f"   Test:  {len(test_data)}")

    # Save
    print("\n💾 Saving datasets...")

    with open(output_dir / "medical_train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / "medical_val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(output_dir / "medical_test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✓ Saved to {output_dir}/")
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
