import telebot
from telebot import types
from model import predict_genre, get_top_songs, convert_to_wav
from config import TELEGRAM_API
import os

tb = telebot.TeleBot(TELEGRAM_API)

user_states = {}

def show_main_menu(chat_id):
    markup = types.ReplyKeyboardMarkup(row_width=2, selective=True)
    itembtn1 = types.KeyboardButton('send song')
    itembtn2 = types.KeyboardButton('most popular')
    markup.add(itembtn1, itembtn2)
    tb.send_message(chat_id, "сhoose one option", reply_markup=markup)

@tb.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    tb.reply_to(message, "Welcome!")
    show_main_menu(chat_id)

@tb.message_handler(func=lambda message: True)
def handle_text(message):
    chat_id = message.chat.id
    text = message.text.lower()

    if text == 'send song':
        tb.send_message(chat_id, "send me the song file.")
        user_states[chat_id] = 'waiting_for_song'
    elif text == 'most popular':
        tb.send_message(chat_id, "enter the genre you are interested in")
        user_states[chat_id] = 'waiting_for_genre'
    elif chat_id in user_states and user_states[chat_id] == 'waiting_for_genre':
        genre = message.text.strip()
        top_tracks = get_top_songs(genre)
        if top_tracks:
            markup = types.InlineKeyboardMarkup()
            for track in top_tracks:
                button = types.InlineKeyboardButton(f"{track['name']} by {track['artist']}", url=track['url'])
                markup.add(button)
            tb.send_message(chat_id, f"top {len(top_tracks)} tracks in genre '{genre}':", reply_markup=markup)
            show_main_menu(chat_id)
        else:
            tb.send_message(chat_id, f"no tracks found for genre '{genre}'.")
        user_states.pop(chat_id, None)
    elif text == 'menu':
        show_main_menu(chat_id)
    else:
        tb.send_message(chat_id, "invalid option.")

@tb.message_handler(content_types=['audio'])
def handle_audio(message):
    chat_id = message.chat.id
    if chat_id in user_states and user_states[chat_id] == 'waiting_for_song':
        file_info = tb.get_file(message.audio.file_id)
        downloaded_file = tb.download_file(file_info.file_path)
        file_extension = os.path.splitext(message.audio.file_name)[-1].lower()
        file_name = message.audio.file_name if message.audio.file_name else "music_file" + file_extension
        with open(file_name, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        if file_extension != '.wav':
            tb.send_message(chat_id, "converting...")
            wav_path = convert_to_wav(file_name)
            if wav_path:
                file_name = wav_path
                tb.send_message(chat_id, "file successfully converted to WAV format.")
            else:
                tb.send_message(chat_id, "failed to convert file to WAV format.")
                return

        predicted_genre = predict_genre(file_name)
        tb.send_message(chat_id, f"The predicted genre is: {predicted_genre}")
        
        similar_tracks = get_top_songs(predicted_genre)
        if similar_tracks:
            markup = types.InlineKeyboardMarkup()
            for track in similar_tracks:
                button = types.InlineKeyboardButton(f"{track['name']} by {track['artist']}", url=track['url'])
                markup.add(button)
            tb.send_message(chat_id, f"Top {len(similar_tracks)} similar tracks for genre '{predicted_genre}':", reply_markup=markup)
        else:
            tb.send_message(chat_id, f"No similar tracks found for genre '{predicted_genre}'.")

        user_states.pop(chat_id, None)
    else:
        tb.reply_to(message, "Please choose 'send song' option first.")

tb.polling()
