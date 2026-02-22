import re
import numpy as np

# Feature order (24 dimensions):
# 0: consonant
# 1: vowel
# 2: labial
# 3: alveolar
# 4: retroflex
# 5: palatal
# 6: velar
# 7: glottal
# 8: stop
# 9: affricate
# 10: fricative
# 11: nasal
# 12: lateral
# 13: approximant
# 14: aspirated
# 15: high
# 16: mid
# 17: low
# 18: front
# 19: central
# 20: back
# 21: rounded
# 22: syllabic_nasal
# 23: rhotic

FEATURE_NAMES = [
    'consonant', 'vowel', 'labial', 'alveolar', 'retroflex', 'palatal', 'velar', 'glottal',
    'stop', 'affricate', 'fricative', 'nasal', 'lateral', 'approximant', 'aspirated',
    'high', 'mid', 'low', 'front', 'central', 'back', 'rounded', 'syllabic_nasal', 'rhotic'
]

# Consonants
CONSONANT_FEATURES = {
    'b':  [1,0, 1,0,0,0,0,0, 1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'p':  [1,0, 1,0,0,0,0,0, 1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0],
    'm':  [1,0, 1,0,0,0,0,0, 0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0],
    'f':  [1,0, 1,0,0,0,0,0, 0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'd':  [1,0, 0,1,0,0,0,0, 1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    't':  [1,0, 0,1,0,0,0,0, 1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0],
    'n':  [1,0, 0,1,0,0,0,0, 0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0],
    'l':  [1,0, 0,1,0,0,0,0, 0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,0],
    'z':  [1,0, 0,1,0,0,0,0, 0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'c':  [1,0, 0,1,0,0,0,0, 0,1,0,0,0,0,1, 0,0,0,0,0,0,0,0,0],
    's':  [1,0, 0,1,0,0,0,0, 0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'zh': [1,0, 0,0,1,0,0,0, 0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'ch': [1,0, 0,0,1,0,0,0, 0,1,0,0,0,0,1, 0,0,0,0,0,0,0,0,0],
    'sh': [1,0, 0,0,1,0,0,0, 0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'r':  [1,0, 0,0,1,0,0,0, 0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,1],
    'j':  [1,0, 0,0,0,1,0,0, 0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'q':  [1,0, 0,0,0,1,0,0, 0,1,0,0,0,0,1, 0,0,0,0,0,0,0,0,0],
    'x':  [1,0, 0,0,0,1,0,0, 0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'g':  [1,0, 0,0,0,0,1,0, 1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
    'k':  [1,0, 0,0,0,0,1,0, 1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0],
    'h':  [1,0, 0,0,0,0,0,1, 0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0],
}

# Vowels/nuclei (monophthongs and diphthong components)
VOWEL_FEATURES = {
    'a':  [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0,0],  # low central
    'aa': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0,0],  # low central
    'e':  [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,0,1,0,0,0],  # mid back unrounded (ê/schwa-like)
    'ee': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,0,1,0,0,0],  # mid back
    'i':  [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # high front
    'ii': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # high front
    'iy': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # high front (yi)
    'ix': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,1,0,0,0,0],  # high central (zhi/chi/shi/ri)
    'iz': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,1,0,0,0,0],  # high central (zi/ci/si)
    'o':  [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,0,1,1,0,0],  # mid back rounded
    'oo': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,0,1,1,0,0],  # mid back rounded
    'u':  [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # high back rounded
    'uu': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # high back rounded
    'v':  [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0],  # high front rounded (ü)
    'vv': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0],  # high front rounded (ü)
    'er': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,1,0,0,0,1],  # mid central rhotic
}

# Compound finals (diphthongs, triphthongs, nasal codas)
COMPOUND_FEATURES = {
    # Diphthongs
    'ai': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0,0],  # a->i, use initial
    'ao': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0,0],  # a->o
    'ei': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,1,0,0,0,0,0],  # e->i
    'ou': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,0,1,1,0,0],  # o->u
    'ia': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i->a
    'ie': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i->e
    'iu': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i->ou
    'ua': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # u->a
    'uo': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # u->o
    'ui': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # u->ei
    've': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0],  # ü->e
    'uai':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # u->ai
    'iao':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i->ao
    
    # Nasal codas
    'an': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0,0],  # a+n
    'en': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,1,0,0,0,0],  # schwa+n
    'in': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i+n
    'un': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # u+n
    'vn': [0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0],  # ü+n
    'ang':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0,0],  # a+ng
    'eng':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,1,0,0,0,0],  # schwa+ng
    'ing':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i+ng
    'ong':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,1,0,0,0,1,1,0,0],  # u+ng
    
    # Complex nasals
    'ian':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0],  # i+a+n
    'uan':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0],  # u+a+n
    'van':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0],  # ü+a+n
    'iang':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0], # i+a+ng
    'uang':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0], # u+a+ng
    'iong':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0], # i+ong
    'ueng':[0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,1,1,0,0], # u+eng
}

# Silence
SILENCE_FEATURES = {
    'sil': [0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],
}

# Merge all
PHONE_FEATURES = {}
PHONE_FEATURES.update(SILENCE_FEATURES)
PHONE_FEATURES.update(CONSONANT_FEATURES)
PHONE_FEATURES.update(VOWEL_FEATURES)
PHONE_FEATURES.update(COMPOUND_FEATURES)


def strip_tone(phone):
    """Remove tone markers (1-5) from phone."""
    return re.sub(r'[1-5]$', '', phone)


def get_unique_phones(lexicon_unique_path):
    """Get unique tone-stripped phones from lexicon."""
    phones = set()
    with open(lexicon_unique_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                phones.add(strip_tone(parts[0]))
    return sorted(phones)


def get_feature_vector(phone):
    """Get articulatory feature vector for a phone."""
    phone = strip_tone(phone)
    if phone in PHONE_FEATURES:
        return np.array(PHONE_FEATURES[phone], dtype=np.float32)
    else:
        print(f"Warning: Unknown phone '{phone}', returning zeros")
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)


def build_feature_matrix(lexicon_unique_path):
    """Build feature matrix for all phones in lexicon."""
    phones = get_unique_phones(lexicon_unique_path)
    phone2idx = {p: i for i, p in enumerate(phones)}
    matrix = np.zeros((len(phones), len(FEATURE_NAMES)), dtype=np.float32)
    
    for phone in phones:
        matrix[phone2idx[phone]] = get_feature_vector(phone)
    
    return matrix, phone2idx, phones


def print_coverage(lexicon_unique_path):
    """Print coverage statistics."""
    phones = get_unique_phones(lexicon_unique_path)
    covered = [p for p in phones if p in PHONE_FEATURES]
    missing = [p for p in phones if p not in PHONE_FEATURES]
    
    print(f"Total unique phones (tone-stripped): {len(phones)}")
    print(f"Covered: {len(covered)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print(f"Missing phones: {missing}")
    
    return covered, missing


def save_glove_format(output_path, phones, matrix):
    """
    Save feature vectors in GloVe format.
    
    GloVe format: each line is "word dim1 dim2 dim3 ..."
    Values are space-separated floats.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, phone in enumerate(phones):
            vec_str = ' '.join(f'{v:.6f}' for v in matrix[i])
            f.write(f'{phone} {vec_str}\n')
    print(f"Saved {len(phones)} vectors to {output_path}")


if __name__ == '__main__':
    covered, missing = print_coverage(path:="doc/lexicon_unique.txt")
    print(f"\nFeature dimensions: {len(FEATURE_NAMES)}")
    print(f"Features: {FEATURE_NAMES}")
    
    matrix, phone2idx, phones = build_feature_matrix(path)
    print(f"\nFeature matrix shape: {matrix.shape}")
    
    # Example
    print("\nExample feature vectors:")
    for p in ['b', 'j', 'a', 'i', 'u', 'v', 'an', 'ing', 'er']:
        if p in phone2idx:
            vec = matrix[phone2idx[p]]
            active = [FEATURE_NAMES[i] for i, v in enumerate(vec) if v == 1]
            print(f"  {p}: {active}")
    
    # Save in GloVe format
    print()
    save_glove_format(output_path:="vectors/baseline.txt", phones, matrix)
