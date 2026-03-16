#!/usr/bin/env python
"""Multinomial Logistic Regression for Tone - p(y|S), p(y|P), p(y|S,P)."""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_segmental_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings


# --- Tone-letter handling + IPA->seg inventory mapping ---

TONE_LETTERS = set("˥˦˧˨˩")

# Map tone-letter sequences to Mandarin tone categories 1-4.
# Your f0 symbols show these four patterns:
# ˥˩ (falling) -> 4
# ˧˥ (rising)  -> 2
# ˨˩˦ (dipping)-> 3
# ˩ (low)      -> 1  (your data uses low level for tone 1; adjust if needed)
TONELETTER_TO_TONE = {
    "˥˩": 4,
    "˧˥": 2,
    "˨˩˦": 3,
    "˩": 1,
}

# Minimal IPA nucleus mapping to your seg-vectors inventory.
# Extend this dict as you encounter more bases.
IPA_BASE_TO_SEG = {
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
    "aj": "ai",
    "ej": "ei",
    "ow": "ou",
    "aw": "ao",   # if your system expects aw->ou, change to "ou"
}

def split_phone_and_tone(phone: str):
    """
    Supports either digit tones (a1..a4) or IPA tone letters (a˥˩ etc).
    Returns (base, tone_int_1to4) or (base, None) if unknown.
    """
    if phone is None:
        return None, None

    # 1) Digit tone at end
    m = re.search(r'([1-4])$', phone)
    if m:
        tone = int(m.group(1))
        base = re.sub(r'[1-5]$', '', phone)  # keep your original strip range
        return base, tone

    # 2) IPA tone letters at end (possibly multiple)
    i = len(phone)
    while i > 0 and phone[i-1] in TONE_LETTERS:
        i -= 1
    base = phone[:i]
    tone_mark = phone[i:]
    if tone_mark:
        tone = TONELETTER_TO_TONE.get(tone_mark)
        return base, tone

    return phone, None

def extract_tone(phone):
    _, tone = split_phone_and_tone(phone)
    return tone

def strip_tone(phone):
    base, _ = split_phone_and_tone(phone)
    return base

def map_base_to_seg(base: str):
    """
    Map a base symbol from the f0 JSON inventory to a key in seg-vectors.
    Falls back to itself if already in seg inventory later (caller can check).
    """
    return IPA_BASE_TO_SEG.get(base, base)


def resample_f0(contour, target_length=20):
    if not contour:
        return np.zeros(target_length, dtype=np.float32)
    f0 = np.array(contour, dtype=np.float32)
    voiced = f0 > 0
    if not np.any(voiced):
        return np.zeros(target_length, dtype=np.float32)
    if not np.all(voiced):
        indices = np.arange(len(f0))
        f0 = np.interp(indices, indices[voiced], f0[voiced])
    return np.interp(np.linspace(0, len(f0)-1, target_length), np.arange(len(f0)), f0).astype(np.float32)


def load_samples(f0_files, seg_emb, f0_length=20):
    """Load samples from F0 JSON files."""
    seg_dim = len(next(iter(seg_emb.values())))
    X_seg, X_pros, y = [], [], []
    
    for path in f0_files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        for char_data in data:
            for vowel in char_data.get('vowels', []):
                tone = extract_tone(vowel['phone'])
                if tone is None:
                    continue             
                # Extract the F0 dictionary (using the script's current expected schema)
                f0_data = vowel.get('f0', {}) 
                contour = f0_data.get('contour', [])
                # --- EXCLUSION LOGIC ---
                # Skip if 'mean' is null, contour is missing, or contour is completely unvoiced (all 0.0)
                if f0_data.get('mean') is None or not contour or not any(v > 0 for v in contour):
                    continue
                base_ipa = strip_tone(vowel['phone'])
                base = map_base_to_seg(base_ipa)
                X_seg.append(seg_emb.get(base, np.zeros(seg_dim, dtype=np.float32)))
                X_pros.append(resample_f0(contour, f0_length))
                y.append(tone - 1)
    
    return np.array(X_seg), np.array(X_pros), np.array(y)


def get_f0_files_by_speaker(f0_dir):
    speaker_files = defaultdict(list)
    for path in Path(f0_dir).rglob('*.json'):
        match = re.search(r'(S\d+)', str(path))
        if match:
            speaker_files[match.group(1)].append(path)
    return speaker_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f0-dir', default='f0')
    parser.add_argument('--seg-vectors', default='vectors/baseline.txt')
    parser.add_argument('--f0-length', type=int, default=20)
    parser.add_argument('--C-values', type=float, nargs='+', default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/regression_results.json')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    seg_emb = load_segmental_embeddings(args.seg_vectors)
    speaker_files = get_f0_files_by_speaker(args.f0_dir)
    speakers = list(speaker_files.keys())
    np.random.shuffle(speakers)
    
    n_train, n_dev = int(len(speakers) * 0.8), int(len(speakers) * 0.1)
    train_files = [f for s in speakers[:n_train] for f in speaker_files[s]]
    dev_files = [f for s in speakers[n_train:n_train+n_dev] for f in speaker_files[s]]
    test_files = [f for s in speakers[n_train+n_dev:] for f in speaker_files[s]]
    
    logger.info(f"Loading data...")
    X_seg_train, X_pros_train, y_train = load_samples(train_files, seg_emb, args.f0_length)
    X_seg_dev, X_pros_dev, y_dev = load_samples(dev_files, seg_emb, args.f0_length)
    X_seg_test, X_pros_test, y_test = load_samples(test_files, seg_emb, args.f0_length)
    
    logger.info(f"Samples: train={len(y_train)}, dev={len(y_dev)}, test={len(y_test)}")
    
    results = {}
    conditions = [
        ('segmental', X_seg_train, X_seg_dev, X_seg_test),
        ('prosodic', X_pros_train, X_pros_dev, X_pros_test),
        ('combined', np.hstack([X_seg_train, X_pros_train]), 
                     np.hstack([X_seg_dev, X_pros_dev]), 
                     np.hstack([X_seg_test, X_pros_test]))
    ]
    
    for name, X_train, X_dev, X_test in conditions:
        logger.info(f"\n{'='*40}\nTraining {name.upper()} model\n{'='*40}")
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_dev_s, X_test_s = scaler.transform(X_dev), scaler.transform(X_test)
        
        # Select best C on dev set
        best_C, best_loss = None, float('inf')
        for C in args.C_values:
            model = LogisticRegression(solver='lbfgs', C=C, max_iter=1000, random_state=args.seed)
            model.fit(X_train_s, y_train)
            dev_loss = log_loss(y_dev, model.predict_proba(X_dev_s))
            if dev_loss < best_loss:
                best_C, best_loss = C, dev_loss
        
        # Train final model
        model = LogisticRegression(solver='lbfgs', C=best_C, max_iter=1000, random_state=args.seed)
        model.fit(X_train_s, y_train)
        
        test_probs = model.predict_proba(X_test_s)
        nll = log_loss(y_test, test_probs)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        
        results[name] = {'nll': nll, 'accuracy': acc, 'best_C': best_C}
        logger.info(f"[{name.upper()}] NLL={nll:.4f}, Acc={acc:.4f}, C={best_C}")
    
    # Summary
    logger.info(f"\n{'='*40}\nSUMMARY\n{'='*40}")
    for name, r in results.items():
        logger.info(f"{name:12} | NLL: {r['nll']:.4f} | Acc: {r['accuracy']:.4f}")
    
    if all(k in results for k in ['segmental', 'prosodic', 'combined']):
        h_s, h_p, h_sp = results['segmental']['nll'], results['prosodic']['nll'], results['combined']['nll']
        logger.info(f"\nI(Y;P|S) = {h_s - h_sp:.4f} nats (prosody beyond segmental)")
        logger.info(f"I(Y;S|P) = {h_p - h_sp:.4f} nats (segmental beyond prosody)")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()