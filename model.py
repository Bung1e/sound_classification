import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
from keras.regularizers import l2


data = pd.read_csv('new.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def load_mel_spectrogram(file_path):
    return np.load(file_path)

le = LabelEncoder()
train_labels = le.fit_transform(train_data['label'])
test_labels = le.transform(test_data['label'])
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

train_mels = np.array([load_mel_spectrogram(f) for f in train_data['melspectrogram_data']])
test_mels = np.array([load_mel_spectrogram(f) for f in test_data['melspectrogram_data']])

train_mels = train_mels / np.max(train_mels)
test_mels = test_mels / np.max(test_mels)

train_mels = train_mels[..., np.newaxis]
test_mels = test_mels[..., np.newaxis]

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 1293, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax', kernel_regularizer=l2(0.01))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_mels, train_labels, epochs=15, validation_data=(test_mels, test_labels))

score = model.evaluate(test_mels, test_labels)
print('Test accuracy:', score[1])

model.save('genre_classification_model.h5')

model.save_weights('genre_classification_checkpoint/')
