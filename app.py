"""Веб-інтерфейс (Flask): форма введення даних клієнта + API /api/predict."""
from flask import Flask, jsonify, render_template, request

from src.predict import predict_customer

app = Flask(__name__)

# (ім'я, мітка, тип, значення за замовчуванням, обов'язкове)
FIELDS = [
    ("is_tv_subscriber", "Підписка на ТБ (0/1)", "int", 1, True),
    ("is_movie_package_subscriber", "Пакет фільмів (0/1)", "int", 0, True),
    ("subscription_age", "Вік підписки (роки)", "float", 2.5, True),
    ("bill_avg", "Середній рахунок", "float", 19, True),
    ("reamining_contract", "Залишок контракту (роки; порожньо = немає контракту)", "float", "", False),
    ("service_failure_count", "Кількість збоїв сервісу", "int", 0, True),
    ("download_avg", "Середнє завантаження (ГБ; порожньо = немає даних)", "float", 28, False),
    ("upload_avg", "Середня віддача (ГБ; порожньо = немає даних)", "float", 2, False),
    ("download_over_limit", "Перевищень ліміту завантаження", "int", 0, True),
]


def parse(payload: dict) -> dict:
    out = {}
    for name, label, typ, _, required in FIELDS:
        raw = payload.get(name)
        if raw is None or str(raw).strip() == "":
            if required:
                raise ValueError(f"Поле «{label}» обов'язкове")
            out[name] = None
            continue
        try:
            val = float(str(raw).replace(",", "."))
        except ValueError:
            raise ValueError(f"Поле «{label}» має бути числом")
        if val < 0:
            raise ValueError(f"Поле «{label}» не може бути від'ємним")
        if name in ("is_tv_subscriber", "is_movie_package_subscriber") and val not in (0, 1):
            raise ValueError(f"Поле «{label}» має бути 0 або 1")
        out[name] = val
    return out


@app.get("/")
def index():
    return render_template("index.html", fields=FIELDS)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(predict_customer(parse(payload)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
