# Sound Classifier Telegram Bot

## Overview
Sound Classifier is a Telegram bot that classifies audio files into predefined categories using machine learning. Users can send an audio file, and the bot will return a prediction of what it contains.

## Features
- Accepts audio files in common formats (WAV, MP3, OGG, etc.)
- Uses a trained machine learning model for sound classification
- Provides real-time predictions via Telegram
- Easy to deploy and customize

## Model and Classification
- The bot utilizes a deep learning model trained on a dataset of various sound categories.
- Feature extraction is performed using `Librosa`.
- The model is implemented in `PyTorch/TensorFlow` (specify which one you used).

## License
This project is open-source under the MIT License.
