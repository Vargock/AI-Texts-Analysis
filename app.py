import uuid
from flask import Flask, render_template, request, session
from modules.db import init_db, save_message, get_recent
from modules.hf_helpers import hf_infer
from modules.helpers import detect_lang, MODELS, LABELS, update_history, bar

app = Flask(__name__)
app.secret_key = "seed-string"

init_db()

COLOR_MAP = {
    "POSITIVE": "green",
    "NEGATIVE": "red",
    "NEUTRAL": "orange",
    "ПОЗИТИВНЫЙ": "green",
    "НЕГАТИВНЫЙ": "red",
    "НЕЙТРАЛЬНЫЙ": "orange",
    "Emotion": "blue",
    "Toxicity": "purple"
}


@app.route("/", methods=["GET", "POST"])
def index():
    
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    user_id = session["user_id"]

    result = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            lang = detect_lang(text)
            labels = LABELS[lang]

            # Sentiment
            sentiment_model = MODELS["ru_sentiment"] if lang == "ru" else MODELS["en_sentiment"]
            sentiment_data = hf_infer(sentiment_model, text)
            sentiment_result = None
            if sentiment_data:
                top_raw_label = max(sentiment_data, key=lambda x: x["score"])
                sentiment = labels.get(
                    top_raw_label["label"].lower() if lang == "ru" else top_raw_label["label"],
                    top_raw_label["label"]
                ).upper()
                score = top_raw_label["score"]
                sentiment_result = {
                    "name": sentiment,
                    "score": int(score * 100),
                    "bar": bar(score),
                    "color": COLOR_MAP.get(sentiment, "black")
                }
                save_message(user_id, text, sentiment, score)

            pos_count = update_history(user_id, sentiment_result["name"] if sentiment_result else "")

            # Emotion (EN only)
            emotion_result = None
            if lang == "en":
                emo_data = hf_infer(MODELS["emotion"], text)
                if emo_data:
                    top_emo = max(emo_data, key=lambda x: x["score"])
                    emotion_result = {
                        "name": top_emo["label"].capitalize(),
                        "score": int(top_emo["score"] * 100),
                        "bar": bar(top_emo["score"]),
                        "color": COLOR_MAP["Emotion"]
                    }

            # Toxicity (EN only)
            tox_result = None
            if lang == "en":
                tox_data = hf_infer(MODELS["toxicity"], text)
                if tox_data:
                    top_tox = max(tox_data, key=lambda x: x["score"])
                    tox_result = {
                        "name": top_tox["label"],
                        "score": int(top_tox["score"] * 100),
                        "bar": bar(top_tox["score"]),
                        "color": COLOR_MAP["Toxicity"]
                    }

            result = {
                "sentiment": sentiment_result,
                "emotion": emotion_result,
                "toxicity": tox_result,
                "pos_count": pos_count
            }

    history = get_recent(user_id, 10)
    return render_template("index.html", result=result, history=history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
