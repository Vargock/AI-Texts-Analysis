import os
import time
from collections import deque
import logging
import requests
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters

from db import init_db, save_message, get_recent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sentiment-bot")

TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Models

MODELS = {
    "en_sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "ru_sentiment": "blanchefort/rubert-base-cased-sentiment",
    "emotion": "j-hartmann/emotion-english-distilroberta-base",
    "toxicity": "unitary/toxic-bert"
}

LABELS = {
    "en": {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"},
    "ru": {"negative": "НЕГАТИВНЫЙ", "neutral": "НЕЙТРАЛЬНЫЙ", "positive": "ПОЗИТИВНЫЙ"}
}

USER_HISTORY = {}

# Helper Functions

def detect_lang(text: str) -> str:
    cyr = sum('а' <= c <= 'я' or 'А' <= c <= 'Я' for c in text)
    lat = sum('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in text)
    return "ru" if cyr > lat else "en"


def hf_infer(model: str, text: str):
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": text},
            timeout=15
        )
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list) and out:
            return out
        return out
    except Exception as e:
        log.warning(f"HF inference failed ({model}): {e}")
        return None


def update_history(user_id: int, sentiment: str, max_len=10) -> int:
    history = USER_HISTORY.setdefault(user_id, deque(maxlen=max_len))
    history.append((sentiment, time.time()))
    recent = list(history)[-5:]
    return sum(s in ("POSITIVE", "ПОЗИТИВНЫЙ") for s, _ in recent)


def bar(score: float) -> str:
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled)


# Core Logic

async def analyze(update, context):
    text = update.message.text.strip()
    if not text:
        return

    user_id = update.effective_user.id
    lang = detect_lang(text)

    # Choose sentiment model
    sentiment_model = MODELS["ru_sentiment"] if lang == "ru" else MODELS["en_sentiment"]
    labels = LABELS[lang]

    # HF API call
    sentiment_data = hf_infer(sentiment_model, text)
    if not sentiment_data:
        await update.message.reply_text("⚠️ Couldn't analyze right now, try again.")
        return

    # Normalize output inline
    if isinstance(sentiment_data, dict):
        sentiment_data = [sentiment_data]
    elif isinstance(sentiment_data, list) and len(sentiment_data) == 1 and isinstance(sentiment_data[0], list):
        sentiment_data = sentiment_data[0]

    # Filter invalid entries
    sentiment_data = [item for item in sentiment_data if isinstance(item, dict) and "label" in item and "score" in item]

    if not sentiment_data:
        await update.message.reply_text("⚠️ Couldn't analyze right now, try again.")
        return

    
    top_raw_label = max(sentiment_data, key=lambda x: x["score"])
    sentiment = labels.get(top_raw_label["label"].lower() if lang == "ru" else top_raw_label["label"], top_raw_label["label"]).upper()
    score = top_raw_label["score"]

    save_message(user_id, text, sentiment, score)
    pos_count = update_history(user_id, sentiment)

    lines = []

    if lang == "en":
        lines.append({
            "POSITIVE": "😊 Looks positive!",
            "NEGATIVE": "😞 Sounds negative.",
            "NEUTRAL": "😐 Neutral tone."
        }[sentiment])
        lines.append(f"Sentiment: {sentiment} ({int(score*100)}%) [{bar(score)}]")
        lines.append(f"Positive in last 5 messages: {pos_count}/5")

        # Emotion
        emo_data = hf_infer(MODELS["emotion"], text)
        if isinstance(emo_data, dict):
            emo_data = [emo_data]
        elif isinstance(emo_data, list) and len(emo_data) == 1 and isinstance(emo_data[0], list):
            emo_data = emo_data[0]
        emo_data = [x for x in emo_data if isinstance(x, dict) and "label" in x and "score" in x]
        if emo_data:
            best = max(emo_data, key=lambda x: x["score"])
            lines.append(f"Emotion: {best['label'].capitalize()}")

        # Toxicity
        tox_data = hf_infer(MODELS["toxicity"], text)
        if isinstance(tox_data, dict):
            tox_data = [tox_data]
        elif isinstance(tox_data, list) and len(tox_data) == 1 and isinstance(tox_data[0], list):
            tox_data = tox_data[0]
        tox_data = [x for x in tox_data if isinstance(x, dict) and "label" in x and "score" in x]
        if tox_data:
            t = max(tox_data, key=lambda x: x["score"])
            lines.append(f"Toxicity: {int(t['score']*100)}% ({t['label']})")

    else:
        lines.append({
            "ПОЗИТИВНЫЙ": "😄 Отличный настрой!",
            "НЕГАТИВНЫЙ": "😞 Похоже на негатив.",
            "НЕЙТРАЛЬНЫЙ": "😐 Спокойный тон."
        }[sentiment])
        lines.append(f"Тональность: {sentiment} ({int(score*100)}%) [{bar(score)}]")
        lines.append(f"Позитивных за последние 5: {pos_count}/5")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# Telegram Bot, User Commands

async def start(update, context):
    await update.message.reply_text(
        "👋 Hey! I'm a bot using Hugging Face models.\n"
        "I analyze your messages for sentiment, emotion, and toxicity.\n"
        "Russian is supported but with limited features.\n\nUse /info to learn more."
    )

async def info(update, context):
    await update.message.reply_text(
        "ℹ️ <b>Features:</b>\n"
        "- Sentiment analysis (EN & RU)\n"
        "- Emotion detection (EN only)\n"
        "- Toxicity detection (EN only)\n"
        "- Tracks last messages per user\n"
        "- Confidence bars & emoji insights\n\n"
        "Use /credits for model sources or /mystats for your stats.",
        parse_mode="HTML"
    )

async def credits(update, context):
    await update.message.reply_text(
        "ℹ️ <b>Model Credits:</b>\n"
        "- <a href='https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest'>English Sentiment</a>\n"
        "- <a href='https://huggingface.co/blanchefort/rubert-base-cased-sentiment'>Russian Sentiment</a>\n"
        "- <a href='https://huggingface.co/j-hartmann/emotion-english-distilroberta-base'>Emotion</a>\n"
        "- <a href='https://huggingface.co/unitary/toxic-bert'>Toxicity</a>",
        parse_mode="HTML"
    )

async def mystats(update, _):
    user_id = update.effective_user.id
    history = get_recent(user_id, 50)
    if not history:
        await update.message.reply_text("No messages analyzed yet.")
        return

    sentiments = [s for _, s, _, _ in history]
    total = len(sentiments)
    pos = sum(s in ("POSITIVE", "ПОЗИТИВНЫЙ") for s in sentiments)
    neu = sum(s in ("NEUTRAL", "НЕЙТРАЛЬНЫЙ") for s in sentiments)
    neg = sum(s in ("NEGATIVE", "НЕГАТИВНЫЙ") for s in sentiments)

    await update.message.reply_text(
        f"Your stats:\n"
        f"Positive: {pos} ({pos*100//total}%)\n"
        f"Neutral: {neu} ({neu*100//total}%)\n"
        f"Negative: {neg} ({neg*100//total}%)"
    )


# Run Bot

def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("credits", credits))
    app.add_handler(CommandHandler("mystats", mystats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze))

    log.info("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
