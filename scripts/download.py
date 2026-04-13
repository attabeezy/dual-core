#!/usr/bin/env python3
"""Download data for Twi dual-stream tokenization research.

Two sources:
  - WAXAL aka_asr (google/WaxalNLP): spontaneous Twi/Akan speech transcriptions (~100k)
  - Ghana NLP pristine-twi (ghananlpcommunity/pristine-twi-english): clean formal Twi text (~999k)

Both are saved as JSONL under data/{lang}/ with a unified schema:
  {"id": ..., "transcription": ..., "source": ...}

Usage:
    python scripts/download.py --output data/ --lang akan
    python scripts/download.py --output data/ --limit 10000
"""

import argparse
import json
import sys
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
    print(f"Downloading {config_name}/{split} from google/WaxalNLP (streaming mode)...")
    
    try:
        dataset = load_dataset("google/WaxalNLP", config_name, split=split, streaming=True)
        # Verify the dataset is not empty by trying to get an iterator
        it = iter(dataset)
    except Exception as e:
        print(f"ERROR: Failed to load google/WaxalNLP ({config_name}/{split}). Check your internet and HF access.")
        print(f"Details: {e}")
        return {"file": None, "count": 0, "error": str(e)}

    # Some versions of datasets might need decode(False) if they contain audio
    try:
        dataset = dataset.decode(False)
        dataset = dataset.remove_columns(["audio"])
    except:
        pass

    output_file = output_dir / f"aka_asr_{split}.jsonl"
    count = 0
    print(f"  Writing to {output_file}...")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                record = {
                    "id": item.get("id", f"waxal_{count}"),
                    "transcription": item.get("transcription", item.get("text", "")),
                    "source": "waxal_asr",
                }
                if record["transcription"]:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                
                if limit and count >= limit:
                    break
    except Exception as e:
        print(f"ERROR during iteration of WAXAL {split}: {e}")
        return {"file": str(output_file), "count": count, "error": str(e)}

    if count == 0:
        print(f"WARNING: No samples found for WAXAL {split}. Streaming might have failed.")
        
    return {"file": str(output_file), "count": count}


def download_pristine_twi(output_dir: Path, limit: int | None = None) -> dict:
    """Download Ghana NLP pristine-twi dataset (clean formal Twi text).

    Uses 'ghananlpcommunity/pristine-twi-english' which is the active parallel corpus.
    We extract the Twi side and carve out 5% validation and 5% test.

    Args:
        output_dir: Directory to save data.
        limit: Maximum number of samples to use (across all splits combined).

    Returns:
        Metadata about downloaded splits.
    """
    from datasets import load_dataset

    dataset_id = "ghananlpcommunity/pristine-twi-english"
    print(f"Downloading {dataset_id} (streaming mode)...")
    
    try:
        # This dataset often uses 'train' split
        dataset = load_dataset(dataset_id, split="train", streaming=True)
        it = iter(dataset)
    except Exception as e:
        print(f"ERROR: Failed to load {dataset_id}. Check your internet.")
        print(f"Details: {e}")
        return {}

    # Detect the text field from the first item
    # pristine-twi-english usually has 'tw' or 'twi' or 'text'
    TEXT_FIELD_CANDIDATES = ["tw", "twi", "text", "transcription", "sentence", "content"]
    text_field: str | None = None

    # Buffer all samples (up to limit) then split 90/5/5
    samples: list[str] = []
    print("  Buffering samples and auto-detecting text field...")
    
    try:
        for i, item in enumerate(dataset):
            if text_field is None:
                # Auto-detect on first item
                for candidate in TEXT_FIELD_CANDIDATES:
                    if candidate in item and isinstance(item[candidate], str):
                        text_field = candidate
                        break
                
                # If it's a parallel dataset, it might be in item['translation']['tw']
                if text_field is None and "translation" in item:
                    trans = item["translation"]
                    if "tw" in trans:
                        text_field = "tw"
                        item = trans # Redirect for this iteration
                    elif "twi" in trans:
                        text_field = "twi"
                        item = trans

                if text_field is None:
                    # Fall back to first string-valued key
                    text_field = next(
                        (k for k, v in item.items() if isinstance(v, str)), None
                    )
                
                if text_field is None:
                    print(f"ERROR: Cannot find a text field in {dataset_id}. Keys: {list(item.keys())}")
                    return {}
                    
                print(f"  Detected field: '{text_field}'")

            # Handle the case where we redirected to 'translation' dict
            val = item.get(text_field, "")
            if not val and "translation" in item:
                val = item["translation"].get(text_field, "")

            text = val.strip() if val else ""
            if text:
                samples.append(text)
            
            if limit and len(samples) >= limit:
                break
                
            if len(samples) % 10000 == 0 and len(samples) > 0:
                print(f"  ...Buffered {len(samples)} samples")
                
    except Exception as e:
        print(f"ERROR during buffering of {dataset_id}: {e}")

    if not samples:
        print(f"ERROR: No samples buffered from {dataset_id}. Check dataset structure.")
        return {}

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
    parser.add_argument("--lang", type=str, default="twi", 
                        help="Language code for output directory (default: 'twi', used 'akan' in notes)")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"],
                        help="WAXAL splits to download (pristine-twi is always split 90/5/5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples for WAXAL ASR per split (useful for quick tests)")
    parser.add_argument("--formal-limit", type=int, default=188000,
                        help="Cap total pristine-twi samples (default: 188k to match WAXAL ASR size)")
    args = parser.parse_args()

    # The research uses 'twi' internally, but allows 'akan' for directory mapping consistency
    lang_dir = args.lang
    output_dir = Path(args.output) / lang_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Twi datasets for research (Mapping to '{lang_dir}' directory)...")
    print(f"Target directory: {output_dir}")

    metadata: dict = {"language": lang_dir, "asr": {}, "tts": {}}

    # --- ASR: WAXAL aka_asr ---
    print("\nASR (spontaneous): WAXAL aka_asr")
    for split in args.splits:
        try:
            result = download_waxal_asr(split, output_dir, args.limit)
            metadata["asr"][split] = result
            if "error" not in result:
                print(f"  {split}: {result['count']} samples -> {result['file']}")
        except Exception as e:
            print(f"  {split}: Unexpected Error - {e}")

    # --- Formal text: Ghana NLP pristine-twi-english ---
    print(f"\nFormal text: Ghana NLP pristine-twi-english (capped at {args.formal_limit:,})")
    try:
        pristine_results = download_pristine_twi(output_dir, args.formal_limit)
        if pristine_results:
            metadata["tts"] = pristine_results
        else:
            print("  FAILED: No data retrieved for pristine-twi.")
    except Exception as e:
        print(f"  Unexpected error downloading pristine-twi: {e}")

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_file}")
    
    # Summary of success
    total_asr = sum(s.get("count", 0) for s in metadata["asr"].values())
    total_tts = sum(s.get("count", 0) for s in metadata["tts"].values())
    
    print("\nDownload Summary:")
    print(f"  Total ASR samples: {total_asr:,}")
    print(f"  Total TTS samples: {total_tts:,}")
    
    if total_asr == 0 or total_tts == 0:
        print("\nCRITICAL WARNING: One or more data streams are empty.")
        print("Check your HuggingFace access and internet connection.")
        sys.exit(1)
    
    print("\nDownload complete! You are ready for Phase II.")


if __name__ == "__main__":
    main()
