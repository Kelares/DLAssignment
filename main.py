import torchaudio
import torch
from matplotlib import pyplot as plt
from NeuralNetwork.Numpy import NeuralNetwork as NN



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
    # plot_waveform(waveform, sample_rate)
    # plot_specgram(waveform, sample_rate)
    waveform = waveform / waveform.abs().max()
    waveform = waveform.reshape([-1, 1])
    print(waveform.shape)
    
    nn = NN([waveform.shape[1], waveform.shape[1]//4, 5])
    nn.forward(waveform)
    nn.backward([1,0,0,0,0])
    print(nn.neurons[-1])
    
if __name__ == "__main__":
    main()