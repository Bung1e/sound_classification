import numpy as np
import librosa
from pydub import AudioSegment
def extract_features(audio_path):

    # audio = AudioSegment.from_mp3(audio_path)
    # audio.export("{audio_path}.wav", format="wav")

    y, sr = librosa.load(audio_path, sr=None, duration=30)
    
    features = []
    

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma))
    features.append(np.var(chroma))


    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spectral_centroids))
    features.append(np.var(spectral_centroids))
    

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.var(spectral_bandwidth))
    

    roll_off = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(roll_off))
    features.append(np.var(roll_off))
    

    zero_crossings = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zero_crossings))
    features.append(np.var(zero_crossings))

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    features.append(np.mean(y_harmonic))
    features.append(np.var(y_harmonic))
    
    perceptual_loudness = librosa.feature.rms(y=y)
    features.append(np.mean(perceptual_loudness))
    features.append(np.var(perceptual_loudness))
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(tempo)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.append(np.mean(mfccs[i]))
        features.append(np.var(mfccs[i]))
    
    return features
