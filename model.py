import numpy as np
import librosa
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import CLIENT_ID, CLIENT_SECRET

# data = pd.read_csv('new.csv')

# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# def load_mel_spectrogram(file_path):
#     return np.load(file_path)

# le = LabelEncoder()
# train_labels = le.fit_transform(train_data['label'])
# test_labels = le.transform(test_data['label'])
# train_labels = to_categorical(train_labels, num_classes=10)
# test_labels = to_categorical(test_labels, num_classes=10)

# train_mels = np.array([load_mel_spectrogram(f) for f in train_data['melspectrogram_data']])
# test_mels = np.array([load_mel_spectrogram(f) for f in test_data['melspectrogram_data']])

# train_mels = train_mels / np.max(train_mels)
# test_mels = test_mels / np.max(test_mels)

# train_mels = train_mels[..., np.newaxis]
# test_mels = test_mels[..., np.newaxis]

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 1293, 1)),
#     MaxPooling2D(2, 2),
#     Dropout(0.3),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax', kernel_regularizer=l2(0.01))
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_mels, train_labels, epochs=15, validation_data=(test_mels, test_labels))

# score = model.evaluate(test_mels, test_labels)
# print('Test accuracy:', score[1])

# model = Sequential([
#     Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(128, 1293, 1)),
#     MaxPooling2D(2, 2),
#     Conv2D(16, (5, 5), activation='relu', padding='valid'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(120, activation='relu'),
#     Dense(84, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax', kernel_regularizer=l2(0.01))
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(train_mels, train_labels, epochs=15, validation_data=(test_mels, test_labels))

# score = model.evaluate(test_mels, test_labels)
# print('Test accuracy:', score[1])
# model.save('genre_classification_model.h5')

# model.save_weights('genre_classification_checkpoint/')

import numpy as np
import librosa
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import CLIENT_ID, CLIENT_SECRET
from pydub import AudioSegment
import os
ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
ffprobe_path = "C:/ffmpeg/bin/ffprobe.exe"

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

model = load_model('Vadzim/saved_models/audio_classification.keras')

class_names = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
} 

def features_extractor(filename):
    audio, sample_rate = librosa.load(filename, sr=None, duration=30)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def predict_genre(file_name):
    mfccs_scaled_features = features_extractor(file_name).reshape(1, -1)
    predicted_label = model.predict(mfccs_scaled_features)
    predicted_class = np.argmax(predicted_label, axis=1)[0]
    return class_names[predicted_class]

def get_top_songs(genre, limit=5):
    result = sp.search(q=f'genre:{genre}', type='playlist', limit=1)
    if result['playlists']['items']:
        playlist_id = result['playlists']['items'][0]['id']
        playlist_tracks = sp.playlist_tracks(playlist_id)
        tracks = playlist_tracks['items']
        all_tracks = []
        
        while playlist_tracks['next']:
            playlist_tracks = sp.next(playlist_tracks)
            tracks.extend(playlist_tracks['items'])
        
        for item in tracks:
            track = item['track']
            all_tracks.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'popularity': track['popularity'],
                'url': track['external_urls']['spotify']
            })
        
        top_tracks = sorted(all_tracks, key=lambda x: x['popularity'], reverse=True)[:limit]
        return top_tracks
    else:
        return []

def convert_to_wav(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return None
