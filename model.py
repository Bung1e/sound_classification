import numpy as np
import librosa
from keras.models import load_model

from config import CLIENT_ID, CLIENT_SECRET
from pydub import AudioSegment
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


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
            track = item.get('track')
            if track:
                all_tracks.append({
                    'name': track.get('name', 'Unknown'),
                    'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown Artist',
                    'popularity': track.get('popularity', 0),
                    'url': track['external_urls']['spotify'] if 'external_urls' in track else 'No URL'
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
