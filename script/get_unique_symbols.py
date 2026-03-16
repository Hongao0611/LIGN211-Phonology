# get the unique symbols in f0/*.json files
import json
import re
import argparse
from pathlib import Path
from collections import Counter

TONE_RE = re.compile(r'([1-5])$')   # classifier uses 1-4 for extract, strip uses 1-5

def extract_tone(phone: str):
    m = re.search(r'([1-4])$', phone)
    return int(m.group(1)) if m else None

def strip_tone(phone: str):
    return re.sub(r'[1-5]$', '', phone)

def iter_vowel_phones(f0_path: Path):
    with open(f0_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for char_data in data:
        vowels = char_data.get("vowels") or []
        for v in vowels:
            phone = v.get("phone")
            if not phone:
                continue
            yield phone, v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0-dir", default="f0", help="Directory containing *.json (recursively)")
    ap.add_argument("--match-classifier-filter", action="store_true",
                    help="Apply the same sample exclusion logic as classifier.py (mean!=None, contour present, any voiced)")
    ap.add_argument("--out", default="doc/unique_symbols.json", help="Optional output path (JSON)")
    args = ap.parse_args()

    raw = Counter()
    base = Counter()
    base_no_tone = Counter()
    tones = Counter()
    files = 0

    for p in Path(args.f0_dir).rglob("*.json"):
        files += 1
        try:
            for phone, v in iter_vowel_phones(p):
                if args.match_classifier_filter:
                    f0_data = v.get("f0", {}) or {}
                    contour = f0_data.get("contour", []) or []
                    if f0_data.get("mean") is None or not contour or not any(val > 0 for val in contour):
                        continue
                    if extract_tone(phone) is None:
                        continue

                raw[phone] += 1
                base_sym = strip_tone(phone)
                base[base_sym] += 1
                base_no_tone[base_sym] += 1  # alias; kept in case you add other normalization later

                t = extract_tone(phone)
                if t is not None:
                    tones[t] += 1
        except Exception:
            # skip unreadable/bad JSON files
            continue

    raw_syms = sorted(raw.keys())
    base_syms = sorted(base.keys())

    print(f"Scanned files: {files}")
    print(f"Unique raw phone symbols: {len(raw_syms)}")
    print(f"Unique base symbols (tone-stripped): {len(base_syms)}")
    print(f"Tones seen (1-4): {dict(sorted(tones.items()))}")

    print("\n--- Raw phone symbols (unique) ---")
    for s in raw_syms:
        print(s)

    print("\n--- Base symbols (tone-stripped, unique) ---")
    for s in base_syms:
        print(s)

    if args.out:
        out_obj = {
            "scanned_files": files,
            "unique_raw_count": len(raw_syms),
            "unique_base_count": len(base_syms),
            "tones_seen_1_4": dict(sorted(tones.items())),
            "raw_symbols": raw_syms,
            "base_symbols": base_syms,
            "raw_counts": dict(raw),
            "base_counts": dict(base),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)
        print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()