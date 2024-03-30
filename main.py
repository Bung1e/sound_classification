import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

y, sr = librosa.load('E:\VS Projects\sound_classification\Data\genres_original\pop\pop.00007.wav')

# print('y:', y, '\n')
# print('y shape:', np.shape(y), '\n')
# print('Sample Rate (KHz):', sr, '\n')

# print('Check Len of Audio:', 661794/22050)

audio_file, _ = librosa.effects.trim(y)


# print('Audio File:', audio_file, '\n')
# print('Audio File shape:', np.shape(audio_file))

# zero_crossings = librosa.zero_crossings(audio_file, pad=False)
# positions = [index for index, value in enumerate(zero_crossings) if value]
# print(positions)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)

# y_harm, y_perc = librosa.effects.hpss(audio_file)
# y_test = np.zeros_like(positions)
# plt.figure(figsize = (16, 6))
# plt.plot(y_harm, color = '#A300F9')
# plt.plot(y_perc, color = '#FFB100')
# plt.scatter(positions, y_test, color='green')
# plt.show()

# spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]

# print('Centroids:', spectral_centroids, '\n')
# print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

# frames = range(len(spectral_centroids))

# t = librosa.frames_to_time(frames)

# print('frames:', frames, '\n')
# print('t:', t)

# def normalize(x, axis=0):
#     return sklearn.preprocessing.minmax_scale(x, axis=axis)
# plt.figure(figsize = (16, 6))
# librosa.display.waveshow(audio_file, sr=sr, alpha=0.4, color = '#A300F9')
# plt.plot(t, normalize(spectral_centroids), color='#FFB100')
# plt.show()

# S = librosa.feature.melspectrogram(y=y, sr=sr)
# S_DB = librosa.amplitude_to_db(S, ref=np.max)
# plt.figure(figsize = (16, 6))
# librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis = 'time', y_axis = 'log', cmap = 'cool') # 0 - 80 это децебелы 
# plt.colorbar()
# plt.title("123", fontsize = 23)
# plt.show()


'''spectral_rolloff величина помогает разделить жанры, например метал значение 5000-7000, классика ~2900'''
spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
print(np.mean(spectral_rolloff))

