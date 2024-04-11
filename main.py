import telebot
from telebot import types
from datasets import load_dataset

from model_wrapper import ModelWrapper

"""
get_text_messages - обработка любого текстового сообщения, в том числе того, что отправился при нажатии кнопки.

Методы, реализующие одноименные команды телеграм-боту:
start
help
generate
model_info
set_model
"""

TOKEN = "6537582989:AAHCIJNfllZtrMUyv6gbbXSpkHnwDAutqLM"

bot = telebot.TeleBot(TOKEN)
model_wrapper = ModelWrapper()


@bot.message_handler(commands=['help'])
def help(message):
    help_message = """Доступны следующие команды:
/start старт бота
/set_model выбор модели
/model_info посмотреть какая модель сейчас загружена
/generate сгенерировать текст по контексту (можно использовать без введения команды)
/train_model обучить новую n-граммную модель!!!
Для обучения нужно указать название новой модели (без пробелов) и прикрепить текстовый файл с датасетом
Пример: /train_model <model_name> *прикреплённый файл* (одним сообщением)
"""
    bot.send_message(message.from_user.id, help_message)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.from_user.id, "Привет! Для знакомства с доступными командами введите /help")


@bot.message_handler(commands=['set_model'])
def set_model(message):
    markup = types.InlineKeyboardMarkup(row_width=1)
    for model_name in model_wrapper.get_available_models():
        markup.add(types.InlineKeyboardButton(model_name, callback_data=f"set_model {model_name}"))
    bot.send_message(message.from_user.id, "Выберите модель для генерации:", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
def callback_query_handler(call):
    command, arg = call.data.split()
    if command == "set_model":
        model_name = arg
        bot.send_message(call.message.chat.id, f"Загрузка модели...")
        model_wrapper.load(model_name)
        bot.send_message(call.message.chat.id, f"Выбрана модель {model_name}")


@bot.message_handler(commands=['train_model'])
def train_model(message):
    bot.send_message(message.from_user.id, 'Напиши название новой модели (без пробелов)')
    bot.register_next_step_handler(message, get_model_name)


def get_model_name(message):
    model_wrapper.new_model_name = message.text
    bot.send_message(message.from_user.id,
                     "Прикрепи текстовый файл датасетом или укажи huggingface датасет, например: IgorVolochay/russian_jokes")
    bot.register_next_step_handler(message, train)


def train(message):
    try:
        bot.send_message(message.from_user.id, 'Скачивание файла...')
        if message.document:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            train_corpus = downloaded_file.decode('utf-8').split('\n')
        else:
            dataset_name = message.text
            dataset = load_dataset(dataset_name, split='train')
            print(len(dataset['text']))
            train_corpus = list(dataset['text'])
        print('model_loaded')

        bot.reply_to(message, '\n'.join(train_corpus[:10]))
        bot.send_message(message.from_user.id, 'Обучение новой модели...')

        model_wrapper.train_and_save_model(model_wrapper.new_model_name, train_corpus)
        bot.send_message(message.from_user.id, f'Обучение завершено, выбрана новая модель: {model_wrapper.new_model_name}')
        model_wrapper.load(model_wrapper.new_model_name)

    except Exception as e:
        bot.send_message(message.from_user.id, f'Произошла ошибка: {e}')


@bot.message_handler(commands=['model_info'])
def model_info(message):
    bot.send_message(message.from_user.id, f"Текущая модель: {str(model_wrapper.current_model_name)}")


@bot.message_handler(commands=['generate'])
def generate(message):
    bot.send_message(message.from_user.id,
                     "Введите текст (вопрос, на который нужно ответить, либо текст, который нужно продолжить)")


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    print(f'<{message.text}>')

    status, result = model_wrapper.generate(message.text)
    if status:
        bot.send_message(message.from_user.id, result)
    else:
        bot.send_message(message.from_user.id, f"Проблемы с генерацией, ниже описаны ошибки.\n{result}")


if __name__ == '__main__':
    bot.polling(none_stop=True, interval=0)
