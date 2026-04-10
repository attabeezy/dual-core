#!/usr/bin/env python3
"""Download data for Twi dual-stream tokenization research.

Two sources:
  - WAXAL aka_asr (google/WaxalNLP): spontaneous Twi/Akan speech transcriptions (~100k)
  - Ghana NLP pristine-twi (ghananlpcommunity/pristine-twi): clean formal Twi text (~999k)

Both are saved as JSONL under data/twi/ with a unified schema:
  {"id": ..., "transcription": ..., "source": ...}

Usage:
    python scripts/download.py --output data/
    python scripts/download.py --output data/ --limit 10000
"""

import argparse
import json
from pathlib import Path


def download_waxal_asr(split: str, output_dir: Path, limit: int | None = None) -> dict:
    """Download WAXAL aka_asr split (spontaneous Twi/Akan speech).

    Args:
        split: HuggingFace split name ('train', 'validation', 'test').
        output_dir: Directory to save data.
        limit: Maximum number of samples to download.

    Returns:
        Metadata about downloaded split.
    """
    from datasets import load_dataset

    config_name = "aka_asr"
    print(f"Downloading {config_name}/{split} (streaming mode)...")
    dataset = load_dataset("google/WaxalNLP", config_name, split=split, streaming=True)
    dataset = dataset.decode(False)
    dataset = dataset.remove_columns(["audio"])

    output_file = output_dir / f"aka_asr_{split}.jsonl"
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            record = {
                "id": item["id"],
                "transcription": item.get("transcription", item.get("text", "")),
                "source": "waxal_asr",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break

    return {"file": str(output_file), "count": count}


def download_pristine_twi(output_dir: Path, limit: int | None = None) -> dict:
    """Download Ghana NLP pristine-twi dataset (clean formal Twi text).

    The dataset has a single 'train' split. We carve out 5% validation and
    5% test from it so downstream scripts can use the standard split names.

    Args:
        output_dir: Directory to save data.
        limit: Maximum number of samples to use (across all splits combined).

    Returns:
        Metadata about downloaded splits.
    """
    from datasets import load_dataset

    print("Downloading ghananlpcommunity/pristine-twi (streaming mode)...")
    dataset = load_dataset("ghananlpcommunity/pristine-twi", split="train", streaming=True)

    # Buffer all samples (up to limit) then split 90/5/5
    samples: list[str] = []
    for i, item in enumerate(dataset):
        text = item.get("text", item.get("transcription", "")).strip()
        if text:
            samples.append(text)
        if limit and len(samples) >= limit:
            break

    n = len(samples)
    val_start = int(n * 0.90)
    test_start = int(n * 0.95)

    splits = {
        "train":      samples[:val_start],
        "validation": samples[val_start:test_start],
        "test":       samples[test_start:],
    }

    results: dict[str, dict] = {}
    for split_name, texts in splits.items():
        output_file = output_dir / f"pristine_twi_{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, text in enumerate(texts):
                record = {
                    "id": f"pristine_twi_{split_name}_{idx}",
                    "transcription": text,
                    "source": "pristine_twi",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        results[split_name] = {"file": str(output_file), "count": len(texts)}
        print(f"  {split_name}: {len(texts)} samples -> {output_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Twi ASR + formal text datasets")
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"],
                        help="WAXAL splits to download (pristine-twi is always split 90/5/5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples for WAXAL ASR per split (useful for quick tests)")
    parser.add_argument("--formal-limit", type=int, default=188000,
                        help="Cap total pristine-twi samples (default: 188k to match WAXAL ASR size)")
    args = parser.parse_args()

    output_dir = Path(args.output) / "twi"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Twi datasets...")
    print(f"Target directory: {output_dir}")

    metadata: dict = {"language": "twi", "asr": {}, "tts": {}}

    # --- ASR: WAXAL aka_asr ---
    print("\nASR (spontaneous): WAXAL aka_asr")
    for split in args.splits:
        try:
            result = download_waxal_asr(split, output_dir, args.limit)
            metadata["asr"][split] = result
            print(f"  {split}: {result['count']} samples -> {result['file']}")
        except Exception as e:
            print(f"  {split}: Error - {e}")

    # --- Formal text: Ghana NLP pristine-twi ---
    print(f"\nFormal text: Ghana NLP pristine-twi (capped at {args.formal_limit:,})")
    try:
        pristine_results = download_pristine_twi(output_dir, args.formal_limit)
        metadata["tts"] = pristine_results
    except Exception as e:
        print(f"  Error downloading pristine-twi: {e}")

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_file}")
    print("Download complete!")


if __name__ == "__main__":
    main()
