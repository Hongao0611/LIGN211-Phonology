# debug_lexicon.py
def debug_lexicon(lexicon_path, transcript_path):
    # Load lexicon
    lexicon = {}
    print("=== Loading Lexicon ===")
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10:  # Show first 10 entries
                print(f"  Line {i}: {repr(line.strip())}")
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                phones = parts[1:]
                lexicon[word] = phones
    
    print(f"\nTotal lexicon entries: {len(lexicon)}")
    print(f"Sample entries: {list(lexicon.items())[:5]}")
    
    # Load transcript
    print("\n=== Loading Transcript ===")
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5:  # Show first 5 entries
                print(f"  Line {i}: {repr(line.strip())}")
    
    # Check specific utterance
    utt_id = "BAC009S0002W0122"
    print(f"\n=== Looking for utterance: {utt_id} ===")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            if utt_id in line:
                print(f"Found: {repr(line.strip())}")
                parts = line.strip().split()
                if len(parts) >= 2:
                    words = parts[1:]
                    print(f"Words: {words}")
                    for word in words[:5]:
                        if word in lexicon:
                            print(f"  '{word}' -> {lexicon[word]}")
                        else:
                            print(f"  '{word}' -> NOT FOUND in lexicon")
                            # Try character-by-character
                            for char in word:
                                if char in lexicon:
                                    print(f"    char '{char}' -> {lexicon[char]}")
                                else:
                                    print(f"    char '{char}' -> NOT FOUND")
                break

# Run debug
debug_lexicon("doc/lexicon.txt", "doc/transcript.txt")