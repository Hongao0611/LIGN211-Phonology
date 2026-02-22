#!/usr/bin/env python
"""Tone Classifier Models (MLP) - M_S, M_P, M_S,P comparison."""

import argparse
import json
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_segmental_embeddings(path):
    """Load embeddings from GloVe-format file."""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                embeddings[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return embeddings


def extract_tone(phone):
    """Extract tone (1-4) from phone."""
    match = re.search(r'([1-4])$', phone)
    return int(match.group(1)) if match else None


def strip_tone(phone):
    """Remove tone marker from phone."""
    return re.sub(r'[1-5]$', '', phone)


def resample_f0(contour, target_length=20):
    """Resample F0 contour to fixed length."""
    if not contour or len(contour) == 0:
        return np.zeros(target_length, dtype=np.float32)
    
    f0 = np.array(contour, dtype=np.float32)
    voiced = f0 > 0
    
    if not np.any(voiced):
        return np.zeros(target_length, dtype=np.float32)
    
    # Interpolate unvoiced regions
    if not np.all(voiced):
        indices = np.arange(len(f0))
        f0 = np.interp(indices, indices[voiced], f0[voiced])
    
    # Resample
    new_indices = np.linspace(0, len(f0) - 1, target_length)
    return np.interp(new_indices, np.arange(len(f0)), f0).astype(np.float32)


def normalize_f0(f0):
    """Z-score normalize F0."""
    if np.all(f0 == 0):
        return f0
    mean, std = np.mean(f0), max(np.std(f0), 1e-6)
    return (f0 - mean) / std


class ToneDataset(Dataset):
    """Dataset loading directly from F0 JSON files."""
    
    def __init__(self, f0_files, seg_embeddings, f0_length=20, 
                 include_seg=True, include_pros=True):
        self.samples = []
        self.seg_dim = len(next(iter(seg_embeddings.values())))
        self.f0_length = f0_length
        self.include_seg = include_seg
        self.include_pros = include_pros
        
        for f0_path in f0_files:
            try:
                with open(f0_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                continue
            
            for char_data in data:
                if not char_data.get('vowels'):
                    continue
                
                for vowel in char_data['vowels']:
                    phone = vowel['phone']
                    tone = extract_tone(phone)
                    if tone is None:
                        continue
                    
                    base = strip_tone(phone)
                    seg_vec = seg_embeddings.get(base, np.zeros(self.seg_dim, dtype=np.float32))
                    
                    contour = vowel.get('f0', {}).get('contour', [])
                    f0_vec = normalize_f0(resample_f0(contour, f0_length))
                    
                    self.samples.append({
                        'seg': seg_vec, 'pros': f0_vec, 'tone': tone - 1
                    })
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = []
        if self.include_seg:
            feats.append(s['seg'])
        if self.include_pros:
            feats.append(s['pros'])
        x = np.concatenate(feats) if len(feats) > 1 else feats[0]
        return torch.tensor(x), torch.tensor(s['tone'])
    
    def input_dim(self):
        return (self.seg_dim if self.include_seg else 0) + \
               (self.f0_length if self.include_pros else 0)


class ToneClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 4))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(x)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(x)
    return total_loss / total, correct / total


def get_f0_files_by_speaker(f0_dir):
    """Group F0 files by speaker."""
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
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/classifier_results.json')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seg_emb = load_segmental_embeddings(args.seg_vectors)
    speaker_files = get_f0_files_by_speaker(args.f0_dir)
    speakers = list(speaker_files.keys())
    np.random.shuffle(speakers)
    
    # 80/10/10 speaker-disjoint split
    n_train, n_dev = int(len(speakers) * 0.8), int(len(speakers) * 0.1)
    train_files = [f for s in speakers[:n_train] for f in speaker_files[s]]
    dev_files = [f for s in speakers[n_train:n_train+n_dev] for f in speaker_files[s]]
    test_files = [f for s in speakers[n_train+n_dev:] for f in speaker_files[s]]
    
    logger.info(f"Speakers: {len(speakers)} | Train/Dev/Test files: {len(train_files)}/{len(dev_files)}/{len(test_files)}")
    
    results = {}
    conditions = [('segmental', True, False), ('prosodic', False, True), ('combined', True, True)]
    
    for name, inc_seg, inc_pros in conditions:
        logger.info(f"\n{'='*50}\nTraining {name.upper()} model\n{'='*50}")
        
        train_ds = ToneDataset(train_files, seg_emb, args.f0_length, inc_seg, inc_pros)
        dev_ds = ToneDataset(dev_files, seg_emb, args.f0_length, inc_seg, inc_pros)
        test_ds = ToneDataset(test_files, seg_emb, args.f0_length, inc_seg, inc_pros)
        
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, args.batch_size)
        test_loader = DataLoader(test_ds, args.batch_size)
        
        model = ToneClassifier(train_ds.input_dim(), args.hidden, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_loss, patience_cnt, best_state = float('inf'), 0, None
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}")
            
            if dev_loss < best_loss:
                best_loss, patience_cnt, best_state = dev_loss, 0, model.state_dict().copy()
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results[name] = {'loss': test_loss, 'accuracy': test_acc}
        logger.info(f"[{name.upper()}] Test: loss={test_loss:.4f}, acc={test_acc:.4f}")
    
    # Summary
    logger.info(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    for name, r in results.items():
        logger.info(f"{name:12} | Loss: {r['loss']:.4f} | Acc: {r['accuracy']:.4f}")
    
    if all(k in results for k in ['segmental', 'prosodic', 'combined']):
        h_s, h_p, h_sp = results['segmental']['loss'], results['prosodic']['loss'], results['combined']['loss']
        logger.info(f"\nInfo gain from prosody: {h_s - h_sp:.4f} nats")
        logger.info(f"Info gain from segmental: {h_p - h_sp:.4f} nats")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()