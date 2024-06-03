import telebot
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import CLIENT_ID, CLIENT_SECRET, TELEGRAM_API


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

bot = telebot.TeleBot(TELEGRAM_API)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Send me a music file.")



@bot.message_handler(content_types=['audio'])
def handle_audio(message):
    file_info = bot.get_file(message.audio.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    file_name = message.audio.file_name if message.audio.file_name else "music_file.mp3"
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(message, f"Music file '{file_name}' has been received and saved.")


bot.polling()