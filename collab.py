import os
import pandas as pd

data_dir = "Train"
files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

# Parse metadata from filenames
data = []
for f in files:
    accent = int(f[0])
    gender = f[1]
    data.append((os.path.join(data_dir, f), accent, gender))

df = pd.DataFrame(data, columns=["filepath", "accent", "gender"])
print(df.head())


import torchaudio
import IPython.display as ipd

waveform, sr = torchaudio.load(df.iloc[0]["filepath"])
print("Sample rate:", sr)
print("Waveform shape:", waveform.shape)
ipd.Audio(waveform.numpy(), rate=sr)

# Install audiomentations if not done yet
import torchaudio
import IPython.display as ipd
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, Gain, TimeStretch, ApplyImpulseResponse
import torch

# Pick any row from your dataframe (e.g., index 0)
row = df.iloc[0] #----------------------------------------------row index here!
file_path = row['filepath']

# Load audio
waveform, sr = torchaudio.load(file_path)
waveform = waveform.mean(dim=0)  # Convert to mono
waveform_np = waveform.numpy()

# Define augmentation pipeline with 5 sequential audio transformations
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),  # 1️⃣ Add background noise (simulates real-world conditions)
    PitchShift(min_semitones=-2, max_semitones=2, p=1.0),              # 2️⃣ Shift pitch up/down without changing speed
    Shift(min_shift=-0.2, max_shift=0.2, p=1.0),                       # 3️⃣ Shift audio in time (forward/backward)
    Gain(min_gain_db=-6, max_gain_db=6, p=0.5),                        # 4️⃣ Random volume change (louder/quieter)
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)                    # 5️⃣ Change speed without affecting pitch
])

# Apply augmentation
augmented_np = augmenter(samples=waveform_np, sample_rate=sr)

# Total number of samples
print("Total samples:", len(df))

# Class distribution (accents)
print("\nAccent distribution:")
print(df["accent"].value_counts().sort_index())

# Gender distribution
print("\nGender distribution:")
print(df["gender"].value_counts())