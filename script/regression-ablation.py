#!/usr/bin/env python
"""Multinomial Logistic Regression for Tone - p(y|S), p(y|P), p(y|S,P) with Feature Ablation."""

import argparse
import json
import csv
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

TONELETTER_TO_TONE = {
    "˥˩": 4,
    "˧˥": 2,
    "˨˩˦": 3,
    "˩": 1,
}

IPA_BASE_TO_SEG = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "aj": "ai", "ej": "ei", "ow": "ou", "aw": "ao",
}

def split_phone_and_tone(phone: str):
    if phone is None:
        return None, None
    m = re.search(r'([1-4])$', phone)
    if m:
        tone = int(m.group(1))
        base = re.sub(r'[1-5]$', '', phone)
        return base, tone

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
                f0_data = vowel.get('f0', {}) 
                contour = f0_data.get('contour', [])
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
    
    # Pre-build combined arrays for easy access
    X_comb_train = np.hstack([X_seg_train, X_pros_train])
    X_comb_dev = np.hstack([X_seg_dev, X_pros_dev])
    X_comb_test = np.hstack([X_seg_test, X_pros_test])

    results = {}
    conditions = [
        ('segmental', X_seg_train, X_seg_dev, X_seg_test),
        ('prosodic', X_pros_train, X_pros_dev, X_pros_test),
        ('combined', X_comb_train, X_comb_dev, X_comb_test)
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

    # --- FEATURE ABLATION (DLL) ---
    logger.info(f"\n{'='*40}\nFEATURE ABLATION (DLL)\n{'='*40}")
    
    seg_dim = X_seg_train.shape[1]
    feature_names = [f"Dim_{i}" for i in range(seg_dim)]
    for base, vec in seg_emb.items():
        if np.sum(vec) == 1.0 and np.max(vec) == 1.0:
            idx = np.argmax(vec)
            feature_names[idx] = base

    base_seg_nll = results['segmental']['nll']
    best_C_seg = results['segmental']['best_C']

    base_comb_nll = results['combined']['nll']
    best_C_comb = results['combined']['best_C']

    ablation_csv_data = []

    for i in range(seg_dim):
        feat_name = feature_names[i]

        # 1. Segmental Ablation
        X_train_seg_abl = np.delete(X_seg_train, i, axis=1)
        X_test_seg_abl = np.delete(X_seg_test, i, axis=1)
        
        scaler_seg = StandardScaler()
        X_train_seg_s = scaler_seg.fit_transform(X_train_seg_abl)
        X_test_seg_s = scaler_seg.transform(X_test_seg_abl)
        
        model_seg = LogisticRegression(solver='lbfgs', C=best_C_seg, max_iter=1000, random_state=args.seed)
        model_seg.fit(X_train_seg_s, y_train)
        
        nll_seg_abl = log_loss(y_test, model_seg.predict_proba(X_test_seg_s))
        dll_seg = nll_seg_abl - base_seg_nll

        # 2. Combined Ablation (Ablating ONLY the segmental feature, keeping prosody intact)
        X_train_comb_abl = np.delete(X_comb_train, i, axis=1)
        X_test_comb_abl = np.delete(X_comb_test, i, axis=1)

        scaler_comb = StandardScaler()
        X_train_comb_s = scaler_comb.fit_transform(X_train_comb_abl)
        X_test_comb_s = scaler_comb.transform(X_test_comb_abl)

        model_comb = LogisticRegression(solver='lbfgs', C=best_C_comb, max_iter=1000, random_state=args.seed)
        model_comb.fit(X_train_comb_s, y_train)

        nll_comb_abl = log_loss(y_test, model_comb.predict_proba(X_test_comb_s))
        dll_comb = nll_comb_abl - base_comb_nll
        
        # Store for CSV
        ablation_csv_data.append({
            'Feature': feat_name,
            'DLL_Segmental': dll_seg,
            'DLL_Combined': dll_comb
        })
        
        logger.info(f"Ablated {feat_name:6} | DLL_Seg: {dll_seg:+.4f} | DLL_Comb: {dll_comb:+.4f}")

    # --- SAVE OUTPUTS ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Save base results to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Save ablation results to CSV (replaces .json extension with .csv)
    csv_path = Path(args.output).with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Feature', 'DLL_Segmental', 'DLL_Combined'])
        writer.writeheader()
        writer.writerows(ablation_csv_data)
        
    logger.info(f"\nSaved regression summary to: {args.output}")
    logger.info(f"Saved DLL ablation results to: {csv_path}")


if __name__ == '__main__':
    main()