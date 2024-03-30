import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

import librosa
import librosa.display
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')

y, sr = librosa.load('sound_classification/123.wav')

# print('y:', y, '\n')
# print('y shape:', np.shape(y), '\n')
# print('Sample Rate (KHz):', sr, '\n')

# print('Check Len of Audio:', 661794/22050)

audio_file, _ = librosa.effects.trim(y)


# print('Audio File:', audio_file, '\n')
# print('Audio File shape:', np.shape(audio_file))

zero_crossings = librosa.zero_crossings(audio_file, pad=False)
print(sum(zero_crossings))
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)