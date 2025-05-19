import torchaudio
import torch
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import torch.nn as nn

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()

def main():
    waveform, sample_rate = torchaudio.load('Train/1f_1018.wav')
    print(waveform, sample_rate)
    plot_waveform(waveform, sample_rate)
    plot_specgram(waveform, sample_rate)


    onlyfiles = [f for f in listdir("Train/") if isfile(join("Train/", f))]
    n = len(onlyfiles)
    waveforms = {}
    loudest = 0
    for file_name in onlyfiles:
        waveform, sample_rate = torchaudio.load(f"Train/{file_name}")
        waveforms[file_name] = waveform
        if waveform.abs().max() > loudest:
            loudest = waveform.abs().max()

    for key in waveforms:
        waveforms[key] = np.array(waveforms[key] / loudest)

    train, test, validate = onlyfiles[:round(n*0.8)], onlyfiles[round(n*0.8): round(n*0.9)], onlyfiles[round(n*0.9):]
    
    print(len(train), len(test), len(validate))
    print(validate[:10])

if __name__ == "__main__":
    main()  