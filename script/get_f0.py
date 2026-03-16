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
import textgrid

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


def parse_mfa_textgrid(textgrid_path: str, silence_marks: set = {'', 'sp', 'sil', 'spn'}) -> Tuple[List[Dict], List[AlignedSegment]]:
    """
    Parses an MFA TextGrid file and extracts precise acoustic alignments.
    Returns character-level and phone-level alignments.
    """
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    
    # MFA typically outputs two tiers: 'words' (characters in Mandarin) and 'phones'
    words_tier = tg.getFirst('words')
    phones_tier = tg.getFirst('phones')
    
    char_alignments = []
    for interval in words_tier:
        if interval.mark not in silence_marks:
            char_alignments.append({
                'character': interval.mark,
                'start': interval.minTime,
                'end': interval.maxTime
            })
            
    phone_alignments = []
    for interval in phones_tier:
        if interval.mark not in silence_marks:
            # Clean up MFA phone markers (e.g., stripping tone numbers if needed)
            clean_phone = interval.mark
            phone_alignments.append(AlignedSegment(
                text=clean_phone,
                start=interval.minTime,
                end=interval.maxTime
            ))
            
    return char_alignments, phone_alignments


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


def process_utterance(extractor: AishellF0Extractor, wav_path: str, textgrid_path: str) -> List[Dict]:
    """Process a single utterance using highly accurate MFA TextGrid alignments."""
    
    # Extract F0 contour for the whole audio file
    times, f0_values, duration = extractor.extract_f0(wav_path)
    
    # Parse the acoustic alignments instead of doing proportional math
    char_alignments, phone_alignments = parse_mfa_textgrid(textgrid_path, extractor.silence | {''})
    
    results = []
    
    for char_info in char_alignments:
        char_text = char_info['character']
        char_start = char_info['start']
        char_end = char_info['end']
        
        # Get character-level F0 using exact acoustic bounds
        char_f0 = extractor.get_f0_stats(times, f0_values, char_start, char_end)
        
        # Find all phones that fall within this character's precise time window
        char_phones = [
            p for p in phone_alignments 
            if p.start >= char_start and p.end <= char_end
        ]
        
        vowel_results = []
        phone_strings = []
        
        for p in char_phones:
            phone_strings.append(p.text)
            
            # If it's a vowel, extract the pitch specifically for that vowel's timespan
            if extractor.is_vowel(p.text):
                vowel_f0 = extractor.get_f0_stats(times, f0_values, p.start, p.end)
                vowel_results.append({
                    'phone': p.text,
                    'start': round(p.start, 4),
                    'end': round(p.end, 4),
                    'f0': vowel_f0
                })
        
        results.append({
            'character': char_text,
            'start': round(char_start, 4),
            'end': round(char_end, 4),
            'phones': phone_strings,
            'char_f0': char_f0,
            'vowels': vowel_results
        })
        
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser("Getting F0 value contours by segment using MFA")
    parser.add_argument("-W", "--wav", default="./wav/", help="Input .wav file or directory")
    parser.add_argument("-G", "--textgrids", default="./textgrids/", help="Directory containing MFA .TextGrid files")
    parser.add_argument("-L", "--lexicon", default="./doc/lexicon.txt", help="Lexicon .txt file")
    args = parser.parse_args()
    # Load wav files
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
    extractor = AishellF0Extractor(args.lexicon)
    # Process each wav file
    for wav in tqdm(wavs):
        utt_id = os.path.basename(wav).replace(".wav", "")
        # Find the matching TextGrid file
        # This assumes your textgrids folder mirrors your wav folder structure
        textgrid_path = wav.replace('/wav/', '/textgrids/').replace(".wav", ".TextGrid")
        if not os.path.exists(textgrid_path):
            # If directory structures don't perfectly mirror, try a flat search in the textgrids folder
            textgrid_path = os.path.join(args.textgrids, f"{utt_id}.TextGrid")
            if not os.path.exists(textgrid_path):
                print(f"Warning: No TextGrid found for {utt_id}, skipping...")
                continue
        # === Process with MFA ===
        results = process_utterance(extractor, wav, textgrid_path)
        # === Save ===
        save_path = wav.replace("wav/", "f0/").replace(".wav", ".json")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w+', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()