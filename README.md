# papre
Pytorch Audio Preprocessors.


Having audio feature transforms implemented as layers allows for powerful on-the-fly computation and eliminates the need for storing them.

It's a similar idea to [kapre](https://github.com/keunwoochoi/kapre), where the performance was studied in [this paper](https://arxiv.org/abs/1706.05781).



## Contents
- [Usage](#usage)




## Usage
Plotting the *Spectrogram* and *Melspectrogram* of a signal.
```python
import torch, librosa
from utils import plot_melspectrogram, plot_spectrogram

y, sr = librosa.load(librosa.util.example_audio_file())
y = torch.from_numpy(y).unsqueeze(0)

win_length = 2048
n_fft = 2048
hop_length = 512

plot_spectrogram(y, sr, 'spec.png', torch.hann_window(win_length), hop_length, n_fft)
plot_melspectrogram(y, sr, 'mel_spec.png', torch.hann_window(win_length), hop_length, n_fft)
```

<p align="center">
<img src="plots/spec.png" width="300px"/>
<img src="plots/mel_spec.png" width="300px"/>
</p>


Using a *Spectrogram* layer in a module is then as easy as:
```python
import torch, librosa
from papre import Melspectrogram

class AudioNN(torch.nn.Module):

    def __init__(self, hop, n_fft, n_mels):
        super(AudioNN, self).__init__()
        self.spec = Melspectrogram(hop=hop, n_fft=n_fft, n_mels=n_mels)
        self.lin = torch.nn.Linear(n_mels, 10)

    def forward(self, x):
        x = self.spec(x)
        x = x.view(x.size(0), -1, x.size(3))
        x = x.mean(dim=1)
        x = self.lin(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

y, sr = librosa.load(librosa.util.example_audio_file())

n_fft = 2048
n_mels = 128
hop = 512

sig = torch.from_numpy(y).view(1,1,-1).cuda()

model = AudioNN(hop=hop, n_fft=n_fft, n_mels=n_mels).cuda()
print(model)

>> AudioNN(
>> (spec): Melspectrogram()
>> (lin): Linear(in_features=128, out_features=10, bias=True)
>> )


spec = model(sig)
print(spec.shape, spec.is_cuda)

>> torch.Size([1, 10]) True

```


