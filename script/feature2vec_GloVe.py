#!/usr/bin/env python
"""Feature2Vec: Phonological features to fixed-length static embeddings using GloVe."""

import argparse
import re
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def strip_tone(phone):
    """Remove tone markers (1-5) from phone."""
    return re.sub(r'[1-5]$', '', phone)


def load_lexicon(path):
    """Load lexicon mapping words to phone sequences (tones stripped)."""
    lexicon = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phones = [strip_tone(p) for p in parts[1:]]
                lexicon[parts[0]].append(phones)
    return dict(lexicon)


def load_phone_inventory(path):
    """Load unique phone inventory (tones stripped, merged)."""
    phones = defaultdict(int)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phone = strip_tone(parts[0])
                phones[phone] += int(parts[1])
    return dict(phones)


def load_transcript(path):
    """Load transcript file."""
    utterances = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utterances.append(parts[1:])
    return utterances


def build_cooccurrence_matrix(utterances, lexicon, phone2idx, window_size=5):
    """Build co-occurrence matrix for phones."""
    vocab_size = len(phone2idx)
    cooccur = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    
    for utterance in utterances:
        phones = []
        for word in utterance:
            if word in lexicon:
                phones.extend(lexicon[word][0])
        
        for i, p in enumerate(phones):
            if p not in phone2idx:
                continue
            idx_i = phone2idx[p]
            for j in range(max(0, i - window_size), min(len(phones), i + window_size + 1)):
                if i != j and phones[j] in phone2idx:
                    cooccur[idx_i, phone2idx[phones[j]]] += 1.0 / abs(i - j)
    
    return cooccur


def train_glove(cooccur, dim=64, x_max=100.0, alpha=0.75, lr=0.05, epochs=50):
    """Train GloVe embeddings from co-occurrence matrix."""
    vocab_size = cooccur.shape[0]
    W = (np.random.rand(vocab_size, dim) - 0.5) / dim
    W_ctx = (np.random.rand(vocab_size, dim) - 0.5) / dim
    b, b_ctx = np.zeros(vocab_size), np.zeros(vocab_size)
    
    nonzero = np.argwhere(cooccur > 0)
    weight = lambda x: np.minimum((x / x_max) ** alpha, 1.0)
    
    for epoch in range(epochs):
        loss = 0.0
        np.random.shuffle(nonzero)
        for i, j in nonzero:
            x_ij = cooccur[i, j]
            diff = np.dot(W[i], W_ctx[j]) + b[i] + b_ctx[j] - np.log(x_ij)
            f_x = weight(x_ij)
            loss += f_x * diff ** 2
            grad = lr * f_x * diff
            W[i] -= grad * W_ctx[j]
            W_ctx[j] -= grad * W[i]
            b[i] -= grad
            b_ctx[j] -= grad
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    return W + W_ctx


def save_embeddings(embeddings, idx2phone, path):
    """Save embeddings in word2vec text format."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{len(idx2phone)} {embeddings.shape[1]}\n")
        for idx, phone in idx2phone.items():
            vec = ' '.join(f"{v:.6f}" for v in embeddings[idx])
            f.write(f"{phone} {vec}\n")


def main():
    parser = argparse.ArgumentParser(description='Feature2Vec with GloVe (tone marker removed)')
    parser.add_argument('--transcript', default='doc/transcript.txt')
    parser.add_argument('--lexicon', default='doc/lexicon.txt')
    parser.add_argument('--lexicon-unique', default='doc/lexicon_unique.txt')
    parser.add_argument('--output', default='vectors/GloVe.txt')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.05)
    args = parser.parse_args()
    
    lexicon = load_lexicon(args.lexicon)
    phone_freq = load_phone_inventory(args.lexicon_unique)
    utterances = load_transcript(args.transcript)
    
    phone2idx = {p: i for i, p in enumerate(phone_freq.keys())}
    idx2phone = {i: p for p, i in phone2idx.items()}
    
    logger.info(f"Vocab (tone-stripped): {len(phone2idx)}, Utterances: {len(utterances)}")
    
    cooccur = build_cooccurrence_matrix(utterances, lexicon, phone2idx, args.window)
    embeddings = train_glove(cooccur, args.dim, epochs=args.epochs, lr=args.lr)
    save_embeddings(embeddings, idx2phone, args.output)
    
    logger.info(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
