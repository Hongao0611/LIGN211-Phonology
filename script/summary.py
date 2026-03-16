import os
import glob
import json
import re
from collections import Counter
import pandas as pd

def generate_summary(f0_dir: str):
    """Parses preprocessed JSON files and generates a summary table."""
    
    # Trackers
    total_utterances = 0
    total_characters = 0
    total_vowels = 0
    valid_f0_vowels = 0
    
    tone_counts = Counter()
    unique_phones = set()
    unique_vowels = set()

    # Find all processed JSON files
    json_files = glob.glob(os.path.join(f0_dir, "**/*.json"), recursive=True)
    print(f"Analyzing {len(json_files)} preprocessed files...")

    for file_path in json_files:
        total_utterances += 1
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        for char_data in data:
            total_characters += 1
            
            # Track all phones in this character
            for phone in char_data.get('phones', []):
                unique_phones.add(phone)
                
            # Track vowel specific data
            for vowel in char_data.get('vowels', []):
                total_vowels += 1
                phone_str = vowel['phone']
                unique_vowels.add(phone_str)
                
                # Check if F0 was successfully extracted (voiced)
                if vowel.get('f0', {}).get('mean') is not None:
                    valid_f0_vowels += 1
                
                # Extract tone label (assuming standard 1-5 MFA format at the end)
                tone_match = re.search(r'([1-5])$', phone_str)
                tone = f"Tone {tone_match.group(1)}" if tone_match else "No Tone"
                tone_counts[tone] += 1

    # Format the collected data into a descriptive summary table
    summary_data = {
        "Metric": [
            "Total Utterances",
            "Total Characters (Syllables)",
            "Total Vowel Tokens",
            "Vowels with Valid F0 (Voiced)",
            "Unique Phones Covered",
            "Unique Vowels Covered",
        ] + [f"Distribution: {tone}" for tone in sorted(tone_counts.keys())],
        
        "Count": [
            total_utterances,
            total_characters,
            total_vowels,
            valid_f0_vowels,
            len(unique_phones),
            len(unique_vowels),
        ] + [tone_counts[tone] for tone in sorted(tone_counts.keys())]
    }

    # Calculate percentages for tones and valid F0
    percentages = ["-", "-", "-", f"{(valid_f0_vowels/total_vowels*100):.1f}%" if total_vowels else "-", "-", "-"]
    percentages += [f"{(tone_counts[tone]/total_vowels*100):.1f}%" if total_vowels else "0%" for tone in sorted(tone_counts.keys())]
    
    df = pd.DataFrame(summary_data)
    df["% of Total Vowels"] = percentages

    print("\n" + "="*50)
    print("DATASET PREPROCESSING SUMMARY")
    print("="*50)
    print(df.to_string(index=False))
    
    # Optional: Save to CSV
    # df.to_csv("dataset_summary.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize preprocessed AISHELL F0 JSONs")
    parser.add_argument("-D", "--dir", default="./f0/", help="Directory containing the JSON files")
    args = parser.parse_args()
    
    generate_summary(args.dir)