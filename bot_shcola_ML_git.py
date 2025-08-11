import os
import logging
import requests
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import pytesseract

# Явно указываем путь к Tesseract
pytesseract.pytesseract.tesseract_cmd = 'указываем путь к Tesseract'
# Настройки DeepSeek API
DEEPSEEK_API_URL = "API_URL"
API_KEY = "ВАШ API_KEY"
PROMPT_PATH = "promt/promt_1.txt"
TOKEN = "ВАШ ТОКЕН"

# Настройка логов
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SubjectClassifier:
    def __init__(self):
        self.keyword_db = self._init_keyword_db()
        self.ml_model = self._init_ml_model()

    def _init_keyword_db(self):
        """База ключевых слов для быстрой классификации"""
        return {
            'математика': ['задача', 'уравнение', 'теорема', 'интеграл', 'производная', 'геометрия', 'алгебра'],
            'физика': ['скорость', 'сила', 'энергия', 'закон', 'ускорение', 'маятник', 'квант'],
            'химия': ['реакция', 'молекула', 'элемент', 'формула', 'вещество', 'валентность', 'оксид'],
            'биология': ['клетка', 'организм', 'ген', 'вид', 'эволюция', 'фотосинтез', 'ДНК'],
            'история': ['война', 'год', 'исторический', 'событие', 'дата', 'империя', 'революция'],
            'литература': ['автор', 'книга', 'персонаж', 'роман', 'поэма', 'стихотворение', 'сюжет'],
            'география': ['страна', 'столица', 'карта', 'материк', 'океан', 'река', 'климат']
        }

    def _init_ml_model(self):
        """Инициализация ML-модели"""
        model_path = "subject_classifier.joblib"

        # Создаем демо-датасет если нет модели
        if not os.path.exists(model_path):
            self._create_demo_dataset()
            return self._train_model()

        return joblib.load(model_path)

    def _create_demo_dataset(self):
        """Создание демо-датасета для обучения"""
        data = {
            'text': [
                "Решите уравнение x² - 5x + 6 = 0",
                "Кто автор 'Войны и мира'?",
                "В каком году началась Вторая мировая война?",
                "Какая формула серной кислоты?",
                "Сколько хромосом у человека?",
                "Чему равно ускорение свободного падения?",
                "Какая самая длинная река в мире?"
            ],
            'subject': [
                'математика', 'литература', 'история',
                'химия', 'биология', 'физика', 'география'
            ]
        }
        pd.DataFrame(data).to_csv("subjects_dataset.csv", index=False)

    def _train_model(self):
        """Обучение ML-модели"""
        df = pd.read_csv("subjects_dataset.csv")
        model = make_pipeline(
            TfidfVectorizer(),
            MultinomialNB()
        )
        model.fit(df['text'], df['subject'])
        joblib.dump(model, "subject_classifier.joblib")
        return model

    def classify_with_keywords(self, text: str) -> str:
        """Быстрая классификация по ключевым словам"""
        text_lower = text.lower()
        for subject, keywords in self.keyword_db.items():
            if any(kw in text_lower for kw in keywords):
                return subject
        return ""

    def classify_with_ml(self, text: str) -> str:
        """Классификация с помощью ML-модели"""
        return self.ml_model.predict([text])[0]

    def classify_with_deepseek(self, text: str) -> str:
        """Глубокая классификация через DeepSeek"""
        prompt = (
            "Определи школьный предмет по вопросу. Выбери только один вариант из: "
            "математика, физика, химия, биология, история, литература, география. "
            f"Вопрос: {text}\nОтвет:"
        )
        # Используем упрощенный запрос без системного промта
        return self.ask_deepseek_simple(prompt)

    def detect_subject(self, text: str) -> str:
        """Комбинированный подход к классификации"""
        # 1. Быстрая проверка по ключевым словам
        fast_result = self.classify_with_keywords(text)
        if fast_result:
            return fast_result

        # 2. ML-классификация
        ml_result = self.classify_with_ml(text)

        # 3. Если ML уверен менее чем на 70%, используем DeepSeek
        proba = self.ml_model.predict_proba([text])[0].max()
        if proba < 0.7:
            return self.classify_with_deepseek(text)

        return ml_result

    def ask_deepseek_simple(self, prompt: str) -> str:
        """Упрощенный запрос к DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 200,
            "top_p": 1,
            "stream": False
        }

        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                json=data,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                return ""

            result = response.json()
            if 'choices' not in result or len(result['choices']) == 0:
                return ""

            return result['choices'][0]['message']['content'].strip()
        except:
            return ""


# Инициализация классификатора
subject_classifier = SubjectClassifier()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне фото с задачей, и я помогу её решить с помощью DeepSeek."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        await update.message.reply_text(f"🧠 Анализирую ваш запрос...")

        # Определяем предмет
        subject = subject_classifier.detect_subject(text)
        logger.info(f"Определен предмет: {subject}")

        # Формируем промт с учетом предмета
        prompt = (f"Ты эксперт по {subject}. Дай точный и лаконичный ответ. "
                  f"Вопрос: {text}\nОтвет:")

        # Отправляем в DeepSeek
        response = ask_deepseek(prompt, subject)
        await update.message.reply_text(
            f"📚 Предмет: {subject.capitalize()}\n\n{response}"
        )

    except Exception as e:
        logger.error(f"Text processing error: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка при обработке текста.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🖼️ Обрабатываю изображение...")

        # Скачиваем фото
        photo_file = await update.message.photo[-1].get_file()
        photo_path = f"temp_photo_{update.message.message_id}.jpg"
        await photo_file.download_to_drive(photo_path)

        # Распознаем текст
        text = recognize_text(photo_path)
        os.remove(photo_path)

        if not text.strip():
            await update.message.reply_text("❌ Не удалось распознать текст")
            return

        # Определяем предмет
        subject = subject_classifier.detect_subject(text)
        logger.info(f"Определен предмет: {subject}")

        # Формируем промт с учетом предмета
        prompt = (f"Ты эксперт по {subject}. Дай точный и лаконичный ответ. "
                  f"Вопрос: {text}\nОтвет:")

        # Отправляем в DeepSeek
        response = ask_deepseek(prompt, subject)
        await update.message.reply_text(
            f"📚 Предмет: {subject.capitalize()}\n\n{response}"
        )

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Ошибка обработки")


def recognize_text(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.point(lambda x: 0 if x < 140 else 255)

        return pytesseract.image_to_string(
            img,
            lang='rus+eng',
            config='--psm 6 --oem 3'
        )
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return ""


def ask_deepseek(prompt: str, subject: str = None) -> str:
    try:
        # Загружаем специализированный промт если есть
        if subject:
            prompt_file = f"prompts/{subject}.txt"
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_p": 1,
            "stop": None,
            "stream": False
        }

        response = requests.post(
            DEEPSEEK_API_URL,
            json=data,
            headers=headers,
            timeout=60
        )

        if response.status_code != 200:
            error_msg = f"DeepSeek API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return "⚠️ Ошибка при обращении к сервису. Попробуйте позже."

        result = response.json()
        if 'choices' not in result or len(result['choices']) == 0:
            logger.error(f"Invalid DeepSeek response: {result}")
            return "⚠️ Не удалось получить ответ от DeepSeek."

        return result['choices'][0]['message']['content'].strip()

    except Exception as e:
        logger.error(f"DeepSeek API Error: {e}")
        return "⚠️ Ошибка при обработке запроса. Попробуйте позже."


if __name__ == "__main__":
    # Установите необходимые зависимости при первом запуске
    try:
        import sklearn
    except ImportError:
        logger.info("Устанавливаем scikit-learn...")
        import subprocess

        subprocess.run(["pip", "install", "scikit-learn", "pandas", "joblib"])
        import sklearn

    # Создаем папку для промтов если ее нет
    prompt_dir = os.path.dirname(PROMPT_PATH)
    if prompt_dir and not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)

    # Создаем папку для предметных промтов
    prompts_subj_dir = "prompts"
    if not os.path.exists(prompts_subj_dir):
        os.makedirs(prompts_subj_dir)

    # Создаем файл с промтом по умолчанию
    if not os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write("Ты полезный помощник для решения учебных задач. Отвечай точно и по существу.")

    # Создаем предметные промты (пример для математики)
    math_prompt_path = os.path.join(prompts_subj_dir, "математика.txt")
    if not os.path.exists(math_prompt_path):
        with open(math_prompt_path, "w", encoding="utf-8") as f:
            f.write("Ты эксперт по математике. Решай задачи шаг за шагом, объясняй кратко и понятно.")

    # Создаем промт для истории
    history_prompt_path = os.path.join(prompts_subj_dir, "история.txt")
    if not os.path.exists(history_prompt_path):
        with open(history_prompt_path, "w", encoding="utf-8") as f:
            f.write("Ты эксперт по истории. Давай точные факты с датами и историческим контекстом.")

    # Аналогично для других предметов...

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))

    logger.info("Бот запущен...")
    app.run_polling()