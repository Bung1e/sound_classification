import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
import librosa
import librosa.display
import keras 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
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

def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) 
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

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

model = keras.models.load_model('Vadzim/saved_models/audio_classification.keras')
filename = "2222.wav"
mfccs_scaled_features = features_extractor(filename).reshape(1, -1)
predicted_label = model.predict(mfccs_scaled_features)
predicted_class = np.argmax(predicted_label, axis=1)[0]
predicted_genre = class_names[predicted_class]

print(f"{predicted_genre}")
