
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from papre import amplitude_to_db, spectrogram, create_mel_filter, \
get_spectrogram_axis

def plot_spec_db(S, t, f, fname='spectro.png', y_range=None):
    """
    Heatmap plot of a spectrogram in db units.
    Args:
     * S: numpy.ndarray
       - db units.
       - Shape: (hop, freq_bins)
     * t: numpy.ndarray
       - second units.
       - Shape: (hop,)
     * f: numpy.ndarray
       - Hz units.
       - Shape: (freq_bins,)
     * fname: str 
       - output file name.
       - Default: 'spectro.png'
     * y_range: tuple  
       - explicit range for y axis
       - Shape: (y_min, y_max)
    """
    if isinstance(S, torch.Tensor):
        S = S.numpy()

    if len(S.shape) != 2:
        S = S.mean(0)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    T, F = np.meshgrid(t, f)
    cmap = cm.get_cmap('magma')
    ax.set_facecolor(cmap.colors[0])

    cbar = ax.pcolormesh(T, F, S.T, cmap=cmap)
    ax.set_xlabel('s')
    ax.set_ylabel('Hz')
    if y_range:
        ax.set_ylim(bottom=y_range[0], top=y_range[1])

    fig.colorbar(cbar, format='%+2.0f dB', ax=ax)

    plt.tight_layout()
    plt.savefig(fname, format='png')
    plt.close()



def plot_spectrogram(sig, sr, fname, window=None, hop=None, n_fft=2048, power=1.0):
    """
    Compute and plot spectrogram given a signal and stft parameters.
    """
    sig_length = sig.size(-1)
    t, f = get_spectrogram_axis(sig_length, sr, n_fft, hop)
    t, f = t.numpy(), f.numpy()
    spec_amp = spectrogram(sig, 
        n_fft=n_fft, 
        hop=hop,
        window=window, 
        power=power)

    spec_db = amplitude_to_db(spec_amp, ref=torch.max)

    plot_spec_db(spec_db, t, f, fname)



def plot_melspectrogram(sig, sr, fname, window=None, hop=None, n_fft=2048, power=1.0, **mel_kwargs):
    """
    Compute and plot mel-spectrogram given a signal and stft parameters.
    """
    sig_length = sig.size(-1)
    t, f = get_spectrogram_axis(sig_length, sr, n_fft, hop)
    t, f = t.numpy(), f.numpy()
    spec_amp = spectrogram(sig, 
        n_fft=n_fft, 
        hop=hop,
        window=window, 
        power=power)

    mel_fb, mel_f = create_mel_filter(spec_amp.size(-1), sr, **mel_kwargs)
    mel_spec_amp = torch.matmul(spec_amp, mel_fb)
    mel_spec_db = amplitude_to_db(mel_spec_amp, ref=torch.max)

    plot_spec_db(mel_spec_db, t, mel_f, fname, y_range=(f.min(), f.max()))



