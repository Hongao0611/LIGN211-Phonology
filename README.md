# Information distribution over Mandarin lexical tone

### Research questions
- RQ1: How informative are segmental cues (syllable identity without tones) about tone category?
- RQ2: How much information about tone is provided by prosodic cues given segmental identity?

### Data
[AISHELL-ASR0009-OS1 Open Source Mandarin Speech Corpus](https://www.aishelltech.com/kysjcp)

### Directory structure
**Doc:** Corpus documents and derived summary data
- `transcript.txt` transcripts of all recordings, each line starts with the file's ID and then the words of the sentence (e.g. *BAC009S0663W0430 爸爸 剪短 发*)
- `speaker.txt` gender of speakers, each line represents one speaker (e.g. *0002 M*)
- `lexicon.txt` vocabulary of the dataset, each line represents a word and its syllables (e.g. *奥运会 aa ao4 vv vn4 h ui4*)
- `lexicon_unique.txt` frequency of every segment that occurred in the corpus (e.g. *a1 5222*)

**Wav:** `.wav` format recording files, organized by train/dev/test split and by participant

**F0:** `.json` format F0 contour files, organized by train/dev/test split and by participant

**Script:** scripts for extracting F0, running MLP and regression models
- `debug.py` check if we can find a specific utterance by its ID in the lexicon file provided
- `get_unique_lexicon.py` calculate `lexicon_unique.txt` from `lexicon.txt`
- `get_f0.py` calculate f0 values from the .wav files, drawing from the transcript and automatically running force-alignment
- `feature2vec_baseline.py` represent all segments in `transcript.txt` by one-hot vectors
- `feature2vec_GloVe.py` represent all segments in `transcript.txt` by GolVe vectors
- `classifier.py` train MLP models for tone category classification
- `classifier.py` train LSTM models for tone category classification
- `regression.py` run logistics regression for tone category classification


**Vectors:** calculated vector representations of segments

### Usage
Make sure you have `conda` in your system and run `conda env create -f environment.yml` to set up an environment with necessary dependencies.

Then, run `conda activate mfa_env` to launch the environment.

I recommended using `python script/get_f0.py --help` to see the available parameters for extracting F0 before running the script itself.

Run `python script/feature2vec_baseline.py` to get interpretable, baseline vectors.

Run `python script/feature2vec_Glove.py` to get uninterpretable, GloVe-style vectors.

Run `python script/classifier.py --help` to look at available arguments.
- `python script/classifier.py` by default runs the baseline model
- `python script/classifier.py --seg-vectors vectors/GloVe_24.txt` runs the GloVe-style vector with fixed length of 24, matching the length of the baseline vectors
- `python script/classifier.py --seg-vectors vectors/GloVe.txt` runs the GloVe-style vector with fixed length of 300, matching the length of the GloVe vectors (which we assumes to be the best static representation we can have)

Run `python script/regression.py --help` to look at available arguments.
- `python script/regression.py` by default runs the regression model with baseline, one-hot vectors as predictors
- `python script/regression.py --seg-vectors vectors/GloVe_24.txt` runs the regression model with GloVe-style vectors (length of 24) as predictors
- `python script/regression.py --seg-vectors vectors/GloVe.txt` runs the regression model with GloVe-style vectors (length of 300) as predictors

### TODO
- exclude instances where the contour is all 0; this could have spoiled the preliminary results
- enable F0 sampling at higher rates; I am highly skeptical about the prosodic channel underperforming the segmental channel and my take would be that the segmental information was better represented (though intuitively, I thought prosodic prediction was a much easier task).