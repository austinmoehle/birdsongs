# Birdsong Classification

Framework for birdsong classification using a finetuned Inception-v3 network.

Contains a TensorFlow model for finetuning Inception-v3 on audio spectrogram data.

## Setup
Create a virtualenv, activate then run
```
pip install -r requirements.txt
```

## Audio Preprocessing
Since Inception-v3 requires square (299x299) input images, raw audio must be converted
to mel spectrograms prior to training/evaluation. The file `processing/tools.py`
contains tools for generating spectrograms from WAV/MP3 audio.

## Train
```
python train.py
```

## Evaluate
```
python eval.py
```
