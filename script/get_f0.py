# f0_extractor_aishell.py
import os
import re
import json
import numpy as np
import parselmouth
from parselmouth.praat import call
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm

@dataclass
class AlignedSegment:
    text: str
    start: float
    end: float


class AishellF0Extractor:
    """F0 Extractor designed for AISHELL dataset format."""
    
    def __init__(self, lexicon_path: str):
        self.lexicon = self._load_lexicon(lexicon_path)
        self._init_phone_sets()
        print(f"Loaded lexicon with {len(self.lexicon)} entries")
    
    def _load_lexicon(self, lexicon_path: str) -> Dict[str, List[str]]:
        """Load lexicon file. Keeps last entry for duplicate words."""
        lexicon = {}
        
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    phones = parts[1:]
                    lexicon[word] = phones
        
        return lexicon
    
    def _init_phone_sets(self):
        """Initialize vowel and consonant phone sets for Mandarin."""
        self.vowel_bases = {
            'a', 'o', 'e', 'i', 'u', 'v',
            'aa', 'oo', 'ee', 'ii', 'ix', 'iy', 'iz',
            'ai', 'ei', 'ao', 'ou',
            'ia', 'ie', 'iu', 'io',
            'ua', 'uo', 'ui', 'ue', 've',
            'iao', 'iou', 'uai', 'uei',
            'an', 'en', 'in', 'un', 'vn',
            'ang', 'eng', 'ing', 'ong',
            'ian', 'uan', 'van', 'uen',
            'iang', 'uang', 'iong', 'ueng',
            'er', 'ir',
        }
        
        self.consonants = {
            'b', 'p', 'm', 'f',
            'd', 't', 'n', 'l',
            'g', 'k', 'h',
            'j', 'q', 'x',
            'zh', 'ch', 'sh', 'r',
            'z', 'c', 's',
            'y', 'w',
        }
        
        self.silence = {'sil', 'sp', 'spn', 'SIL', '<SPOKEN_NOISE>'}
    
    def is_vowel(self, phone: str) -> bool:
        """Check if a phone is a vowel (nucleus that carries tone)."""
        if not phone or phone in self.silence:
            return False
        
        phone_base = re.sub(r'[1-5]$', '', phone.lower())
        
        if phone_base in self.vowel_bases:
            return True
        if phone_base in self.consonants:
            return False
        if phone_base and phone_base[0] in 'aeiouv':
            return True
        
        return False
    
    def get_phones_for_character(self, char: str) -> List[str]:
        """Get phone sequence for a single character."""
        if char in self.lexicon:
            return self.lexicon[char]
        return ['<unk>']
    
    def split_word_to_characters(self, word: str) -> List[Tuple[str, List[str]]]:
        """
        Split a word into characters with their phones.
        Returns list of (character, phones) tuples.
        """
        # If the whole word is in lexicon and has multiple characters,
        # distribute phones across characters
        if word in self.lexicon and len(word) > 1:
            word_phones = self.lexicon[word]
            return self._distribute_phones_to_chars(word, word_phones)
        
        # Otherwise, look up each character individually
        result = []
        for char in word:
            phones = self.get_phones_for_character(char)
            result.append((char, phones))
        return result
    
    def _distribute_phones_to_chars(self, word: str, phones: List[str]) -> List[Tuple[str, List[str]]]:
        """
        Distribute a phone sequence across characters.
        Each Chinese character = one syllable = (optional consonant + vowel).
        """
        result = []
        phone_idx = 0
        
        for char in word:
            char_phones = []
            
            # Collect phones for this character (typically consonant + vowel)
            while phone_idx < len(phones):
                p = phones[phone_idx]
                
                if not char_phones:
                    # First phone for this character
                    char_phones.append(p)
                    phone_idx += 1
                elif self.is_vowel(p):
                    # Vowel follows consonant - part of this syllable
                    char_phones.append(p)
                    phone_idx += 1
                    break  # Vowel marks end of syllable
                else:
                    # Next consonant = next character starts
                    break
            
            result.append((char, char_phones))
        
        return result
    
    def extract_f0(self, wavs: str, time_step: float = 0.01,
                   f0_min: float = 75, f0_max: float = 500) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extract F0 contour from a wav file."""
        sound = parselmouth.Sound(wavs)
        pitch = call(sound, "To Pitch", time_step, f0_min, f0_max)
        
        duration = sound.get_total_duration()
        times = np.arange(0, duration, time_step)
        f0_values = []
        
        for t in times:
            f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
            f0_values.append(f0 if not np.isnan(f0) else 0)
        
        return times, np.array(f0_values), duration
    
    def get_f0_stats(self, times: np.ndarray, f0_values: np.ndarray,
                     start: float, end: float) -> Dict:
        """Get F0 statistics for a time segment."""
        mask = (times >= start) & (times <= end)
        segment_f0 = f0_values[mask]
        voiced_f0 = segment_f0[segment_f0 > 0]
        
        if len(voiced_f0) == 0:
            return {
                'mean': None, 'median': None, 'min': None,
                'max': None, 'std': None, 'contour': segment_f0.tolist()
            }
        
        return {
            'mean': float(np.mean(voiced_f0)),
            'median': float(np.median(voiced_f0)),
            'min': float(np.min(voiced_f0)),
            'max': float(np.max(voiced_f0)),
            'std': float(np.std(voiced_f0)),
            'contour': segment_f0.tolist()
        }


class CharacterLevelAligner:
    """Align at character level (single Chinese character = single syllable)."""
    
    CONSONANT_WEIGHT = 0.3
    VOWEL_WEIGHT = 1.0
    
    def __init__(self, extractor: AishellF0Extractor):
        self.extractor = extractor
    
    def align_characters(self, words: List[str], total_duration: float,
                         start_offset: float = 0.0) -> Tuple[List[AlignedSegment], List[Tuple[str, List[str]]]]:
        """
        Split all words into characters and align each character.
        Returns (alignments, char_phones_list).
        """
        # Flatten words to characters with phones
        all_chars = []
        for word in words:
            char_phones_list = self.extractor.split_word_to_characters(word)
            all_chars.extend(char_phones_list)
        
        if not all_chars:
            return [], []
        
        # Calculate weight for each character
        char_weights = []
        for char, phones in all_chars:
            weight = 0
            for p in phones:
                if p in self.extractor.silence or p == '<unk>':
                    continue
                elif self.extractor.is_vowel(p):
                    weight += self.VOWEL_WEIGHT
                else:
                    weight += self.CONSONANT_WEIGHT
            char_weights.append(max(weight, 0.3))
        
        total_weight = sum(char_weights)
        
        # Distribute time
        aligned = []
        current_time = start_offset
        
        for (char, phones), weight in zip(all_chars, char_weights):
            char_duration = (weight / total_weight) * total_duration
            aligned.append(AlignedSegment(
                text=char,
                start=current_time,
                end=current_time + char_duration
            ))
            current_time += char_duration
        
        return aligned, all_chars
    
    def align_phones(self, phones: List[str], start: float, end: float) -> List[AlignedSegment]:
        """Distribute phones within a character's time span."""
        if not phones:
            return []
        
        phones = [p for p in phones if p not in self.extractor.silence and p != '<unk>']
        if not phones:
            return []
        
        duration = end - start
        phone_weights = []
        
        for p in phones:
            if self.extractor.is_vowel(p):
                phone_weights.append(self.VOWEL_WEIGHT)
            else:
                phone_weights.append(self.CONSONANT_WEIGHT)
        
        total_weight = sum(phone_weights)
        if total_weight == 0:
            return []
        
        aligned = []
        current_time = start
        
        for phone, weight in zip(phones, phone_weights):
            phone_duration = (weight / total_weight) * duration
            aligned.append(AlignedSegment(
                text=phone,
                start=current_time,
                end=current_time + phone_duration
            ))
            current_time += phone_duration
        
        return aligned


def load_aishell_transcript(transcript_path: str) -> Dict[str, List[str]]:
    """Load AISHELL transcript file."""
    transcripts = {}
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id = parts[0]
                text = parts[1]
                words = text.split()
                transcripts[utt_id] = words
            elif len(parts) == 1:
                transcripts[parts[0]] = []
    
    return transcripts


def process_utterance(extractor: AishellF0Extractor, wavs: str,
                      words: List[str]) -> List[Dict]:
    """Process a single utterance at CHARACTER level."""
    
    # Extract F0
    times, f0_values, duration = extractor.extract_f0(wavs)
    
    # Character-level alignment
    aligner = CharacterLevelAligner(extractor)
    char_alignments, char_phones_list = aligner.align_characters(words, duration)
    
    results = []
    
    for char_seg, (char, phones) in zip(char_alignments, char_phones_list):
        # Get character-level F0
        char_f0 = extractor.get_f0_stats(times, f0_values, char_seg.start, char_seg.end)
        
        # Align phones within character
        phone_alignments = aligner.align_phones(phones, char_seg.start, char_seg.end)
        
        # Extract vowel F0
        vowel_results = []
        for phone_seg in phone_alignments:
            if extractor.is_vowel(phone_seg.text):
                vowel_f0 = extractor.get_f0_stats(
                    times, f0_values, phone_seg.start, phone_seg.end
                )
                vowel_results.append({
                    'phone': phone_seg.text,
                    'start': round(phone_seg.start, 4),
                    'end': round(phone_seg.end, 4),
                    'f0': vowel_f0
                })
        
        results.append({
            'character': char,
            'start': round(char_seg.start, 4),
            'end': round(char_seg.end, 4),
            'phones': phones,
            'char_f0': char_f0,
            'vowels': vowel_results
        })
    
    return results


def test():
    # === Configuration ===
    wavs = "wav/train/S0002/BAC009S0002W0122.wav"
    transcript_path = "doc/transcript.txt"
    lexicon_path = "doc/lexicon.txt"
    
    # === Initialize ===
    extractor = AishellF0Extractor(lexicon_path)
    transcripts = load_aishell_transcript(transcript_path)
    
    # Get utterance
    utt_id = os.path.basename(wavs).replace(".wav", "")
    words = transcripts.get(utt_id, [])
    
    print(f"Utterance: {utt_id}")
    print(f"Words: {' '.join(words)}")
    
    # Show character breakdown
    print("\n--- Character Breakdown ---")
    total_chars = 0
    for word in words:
        char_phones = extractor.split_word_to_characters(word)
        for char, phones in char_phones:
            vowels = [p for p in phones if extractor.is_vowel(p)]
            consonants = [p for p in phones if not extractor.is_vowel(p) and p not in extractor.silence]
            print(f"  {char}: consonants={consonants}, vowels={vowels}")
            total_chars += 1
    print(f"Total characters: {total_chars}")
    
    if not words:
        print("ERROR: No transcript found!")
        return
    
    # === Process ===
    results = process_utterance(extractor, wavs, words)
    
    # === Output ===
    print("\n" + "=" * 60)
    print("F0 EXTRACTION RESULTS (Character Level)")
    print("=" * 60)
    
    for r in results:
        print(f"\n【{r['character']}】 ({r['start']:.3f}s - {r['end']:.3f}s)")
        print(f"  Phones: {' '.join(r['phones'])}")
        
        if r['char_f0']['mean'] is not None:
            print(f"  F0: mean={r['char_f0']['mean']:.1f} Hz, "
                  f"range=[{r['char_f0']['min']:.1f}, {r['char_f0']['max']:.1f}] Hz")
        else:
            print("  F0: unvoiced")
        
        for v in r['vowels']:
            if v['f0']['mean'] is not None:
                print(f"    Vowel [{v['phone']}]: mean={v['f0']['mean']:.1f} Hz")
            else:
                print(f"    Vowel [{v['phone']}]: unvoiced")
    
    # Save results
    output_path = f"{utt_id}_f0.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    # === Configuration ===
    import argparse
    parser = argparse.ArgumentParser("Getting F0 value contours by segment")
    parser.add_argument("-W","--wav",default="./wav/",help="Input .wav file or directory containing .wav files")
    parser.add_argument("-T","--transcript",default="./doc/transcript.txt",help="Transcript .txt file")
    parser.add_argument("-L","--lexicon",default="./doc/lexicon.txt",help="Lexicon .txt file")
    args = parser.parse_args()
    
    # load wav files
    if os.path.isfile(args.wav):
        wavs = [args.wav] 
    else:
        wavs = []
        for root, dirs, files in os.walk(args.wav):
            for file in files:
                if file.endswith(".wav"):
                    wavs.append(os.path.join(root, file))
        if not wavs:
            print(f"ERROR: No .wav files found in directory: {args.wav}")
            return
        else:
            print(f"Found {len(wavs)} wav files")
    
    transcript_path = args.transcript
    lexicon_path = args.lexicon
    
    # === Initialize ===
    extractor = AishellF0Extractor(lexicon_path)
    transcripts = load_aishell_transcript(transcript_path)
    
    # Process each wav file
    for wav in tqdm(wavs):
        assert type(wav) == str, f"Expected wav to be a string path:{type(wav), wav}"
        utt_id = os.path.basename(wav).replace(".wav", "")
        words = transcripts.get(utt_id, [])
        
        if not words:
            continue
        
        # === Process ===
        results = process_utterance(extractor, wav, words)
        
        # === Save ===
        save_path = wav.replace("wav/", "f0/").replace(".wav", ".json")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w+', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()