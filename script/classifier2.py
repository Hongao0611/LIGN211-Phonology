#!/usr/bin/env python
"""Tone Classifier Models (Deep Dual-Branch CNN-BiLSTM + ResNet)"""

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

# --- [Data Loading Functions Remain the Same] ---
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
    if not contour or len(contour) == 0:
        return np.zeros(target_length, dtype=np.float32)
    f0 = np.array(contour, dtype=np.float32)
    voiced = f0 > 0
    if not np.any(voiced):
        return np.zeros(target_length, dtype=np.float32)
    if not np.all(voiced):
        indices = np.arange(len(f0))
        f0 = np.interp(indices, indices[voiced], f0[voiced])
    new_indices = np.linspace(0, len(f0) - 1, target_length)
    return np.interp(new_indices, np.arange(len(f0)), f0).astype(np.float32)

def normalize_f0(f0):
    if np.all(f0 == 0):
        return f0
    mean, std = np.mean(f0), max(np.std(f0), 1e-6)
    return (f0 - mean) / std

# --- [Modified Dataset to Return Dicts for Dual-Branch Processing] ---
class ToneDataset(Dataset):
    def __init__(self, f0_files, seg_embeddings, f0_length=20, include_seg=True, include_pros=True):
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
                if not char_data.get('vowels'): continue
                for vowel in char_data['vowels']:
                    phone = vowel['phone']
                    tone = extract_tone(phone)
                    if tone is None: continue
                    
                    f0_data = vowel.get('f0', {})
                    contour = f0_data.get('contour', [])
                    if f0_data.get('mean') is None or not contour or not any(v > 0 for v in contour):
                        continue
                    
                    base_ipa = strip_tone(phone)
                    base = map_base_to_seg(base_ipa)

                    seg_vec = seg_embeddings.get(base)
                    if seg_vec is None:
                        # OOV -> zeros, but log occasionally so you notice mapping gaps
                        seg_vec = np.zeros(self.seg_dim, dtype=np.float32)
                    f0_vec = normalize_f0(resample_f0(contour, f0_length))
                    
                    self.samples.append({'seg': seg_vec, 'pros': f0_vec, 'tone': tone - 1})
        
        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        # Return as a dictionary so the model branches can route data appropriately
        return {
            'seg': torch.tensor(s['seg']) if self.include_seg else torch.empty(0),
            'pros': torch.tensor(s['pros']) if self.include_pros else torch.empty(0),
            'tone': torch.tensor(s['tone'], dtype=torch.long)
        }


# --- [The New Deep Architecture] ---

class ResidualBlock(nn.Module):
    """Deep residual block to prevent vanishing gradients in deep MLPs."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.net(x))

class ProsodyEncoder(nn.Module):
    """CNN-BiLSTM encoder with Attention Pooling for time-series F0 data."""
    def __init__(self, seq_len, out_dim=128, dropout=0.3):
        super().__init__()
        # 1D Conv to extract local pitch variations/derivatives
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # BiLSTM for forward/backward temporal sequence modeling
        self.lstm = nn.LSTM(input_size=16, hidden_size=out_dim//2, 
                            num_layers=2, bidirectional=True, 
                            batch_first=True, dropout=dropout if dropout > 0 else 0)
        
        # Attention mechanism to weight important parts of the pitch contour
        self.attention = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.Tanh(),
            nn.Linear(out_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, channels, seq_len)
        x = x.unsqueeze(1)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2) # (batch, seq_len, channels)
        
        lstm_out, _ = self.lstm(x) # lstm_out: (batch, seq_len, out_dim)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch, out_dim)
        return context

class DeepMultimodalToneClassifier(nn.Module):
    def __init__(self, seg_dim, f0_length, inc_seg, inc_pros, 
                 hidden_dim=128, num_res_blocks=3, dropout=0.3):
        super().__init__()
        self.inc_seg = inc_seg
        self.inc_pros = inc_pros
        
        fusion_dim = 0
        
        # Branch 1: Segmental Processing
        if self.inc_seg:
            self.seg_proj = nn.Sequential(
                nn.Linear(seg_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            self.seg_res_blocks = nn.Sequential(
                *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
            )
            fusion_dim += hidden_dim
            
        # Branch 2: Prosodic Processing
        if self.inc_pros:
            self.pros_encoder = ProsodyEncoder(f0_length, out_dim=hidden_dim, dropout=dropout)
            fusion_dim += hidden_dim
            
        # Branch 3: Late Fusion & Classification
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)],
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4) # 4 tone categories
        )

    def forward(self, batch):
        features = []
        
        if self.inc_seg:
            seg = batch['seg']
            seg_feat = self.seg_res_blocks(self.seg_proj(seg))
            features.append(seg_feat)
            
        if self.inc_pros:
            pros = batch['pros']
            pros_feat = self.pros_encoder(pros)
            features.append(pros_feat)
            
        # Concatenate encoded representations (Late Fusion)
        fused = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.fusion_net(fused)

# --- [Modified Training & Eval Loops] ---

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        # Move required tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch['tone']
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch['tone']
        
        logits = model(batch)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total

def get_f0_files_by_speaker(f0_dir):
    speaker_files = defaultdict(list)
    for path in Path(f0_dir).rglob('*.json'):
        match = re.search(r'(S\d+)', str(path))
        if match: speaker_files[match.group(1)].append(path)
    return speaker_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f0-dir', default='f0')
    parser.add_argument('--seg-vectors', default='vectors/baseline.txt')
    parser.add_argument('--f0-length', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4) # Slightly lower LR for complex models
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/classifier_results.json')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seg_emb = load_segmental_embeddings(args.seg_vectors)
    if not seg_emb:
        # Fallback dummy dim if file missing during testing
        seg_emb = {'dummy': np.zeros(300, dtype=np.float32)} 
        
    speaker_files = get_f0_files_by_speaker(args.f0_dir)
    speakers = list(speaker_files.keys())
    np.random.shuffle(speakers)
    
    n_train, n_dev = int(len(speakers) * 0.8), int(len(speakers) * 0.1)
    train_files = [f for s in speakers[:n_train] for f in speaker_files[s]]
    dev_files = [f for s in speakers[n_train:n_train+n_dev] for f in speaker_files[s]]
    test_files = [f for s in speakers[n_train+n_dev:] for f in speaker_files[s]]
    
    logger.info(f"Speakers: {len(speakers)} | Train/Dev/Test files: {len(train_files)}/{len(dev_files)}/{len(test_files)}")
    
    results = {}
    conditions = [('segmental', True, False), ('prosodic', False, True), ('combined', True, True)]
    
    # We need the seg_dim from the dataset for model initialization
    temp_ds = ToneDataset([], seg_emb)
    seg_dim = temp_ds.seg_dim
    
    for name, inc_seg, inc_pros in conditions:
        logger.info(f"\n{'='*50}\nTraining {name.upper()} model\n{'='*50}")
        
        train_ds = ToneDataset(train_files, seg_emb, args.f0_length, inc_seg, inc_pros)
        dev_ds = ToneDataset(dev_files, seg_emb, args.f0_length, inc_seg, inc_pros)
        test_ds = ToneDataset(test_files, seg_emb, args.f0_length, inc_seg, inc_pros)
        
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, args.batch_size)
        test_loader = DataLoader(test_ds, args.batch_size)
        
        model = DeepMultimodalToneClassifier(
            seg_dim=seg_dim, 
            f0_length=args.f0_length, 
            inc_seg=inc_seg, 
            inc_pros=inc_pros,
            hidden_dim=args.hidden,
            num_res_blocks=args.res_blocks,
            dropout=args.dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_loss, patience_cnt, best_state = float('inf'), 0, None
        for epoch in range(args.epochs):
            if len(train_loader) == 0: break # Guard against empty datasets
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1:03d}: train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}")
            
            if dev_loss < best_loss:
                best_loss, patience_cnt, best_state = dev_loss, 0, model.state_dict().copy()
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
            
        if len(test_loader) > 0:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            results[name] = {'loss': test_loss, 'accuracy': test_acc}
            logger.info(f"[{name.upper()}] Test: loss={test_loss:.4f}, acc={test_acc:.4f}")
    
    logger.info(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    for name, r in results.items():
        logger.info(f"{name:12} | Loss: {r['loss']:.4f} | Acc: {r['accuracy']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()