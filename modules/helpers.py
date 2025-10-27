import time
from collections import deque

# Models and labels
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

# Helpers
def detect_lang(text: str) -> str:
    cyr = sum('а' <= c <= 'я' or 'А' <= c <= 'Я' for c in text)
    lat = sum('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in text)
    return "ru" if cyr > lat else "en"

def update_history(user_id: int, sentiment: str, max_len=10) -> int:
    history = USER_HISTORY.setdefault(user_id, deque(maxlen=max_len))
    history.append((sentiment, time.time()))
    recent = list(history)[-5:]
    return sum(s in ("POSITIVE", "ПОЗИТИВНЫЙ") for s, _ in recent)

def bar(score: float) -> str:
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled)
