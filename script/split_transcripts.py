# split_transcripts.py
import os

transcript_path = "doc/transcript.txt"
wav_dir = "wav/"

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, text = parts
            
            # Find the corresponding wav file to put the txt file next to it
            for root, dirs, files in os.walk(wav_dir):
                if f"{utt_id}.wav" in files:
                    txt_path = os.path.join(root, f"{utt_id}.txt")
                    with open(txt_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(text)
                    break