# Прогнозування відтоку клієнтів інтернет-провайдера

Фінальний проєкт GoIT (Python Data Science and Machine Learning).

## Мета проєкту
Побудувати модель, яка за історичними даними про клієнтів оцінює **ймовірність припинення користування послугами (churn)**, і зробити зручний інтерфейс: ввів дані клієнта — отримав «Клієнт має високу/низьку ймовірність відтоку» та ймовірність у відсотках. Модель запакована в Docker-контейнер для відтворюваності.

Датасет: `data/internet_service_churn.csv` (72 274 клієнти, 9 ознак + `churn`; ≈55% клієнтів пішли).

## Результати
Тестова вибірка (20%, стратифікована), найкраща модель — **HistGradientBoostingClassifier** (відбір за F1 у 5-fold крос-валідації):

| Модель | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **HistGradientBoosting** | 0.9484 | 0.9435 | 0.9566 | 0.9408 | **0.9486** | 0.9826 |
| Random Forest | 0.9452 | 0.9386 | 0.9544 | 0.9337 | 0.9440 | 0.9797 |
| Logistic Regression | 0.9399 | 0.9325 | 0.9504 | 0.9266 | 0.9384 | 0.9609 |

Головний фактор відтоку — залишок контракту (`reamining_contract`); також важливі обсяг завантаження, середній рахунок і наявність ТБ-підписки. Повний аналіз — у `notebooks/churn_analysis.ipynb`.

## Структура репозиторію
```
.
├── data/internet_service_churn.csv   # вихідні дані
├── notebooks/churn_analysis.ipynb    # EDA, обробка, порівняння моделей, аналіз результатів
├── src/
│   ├── preprocessing.py              # інженерія ознак + sklearn Pipeline
│   ├── eda.py                        # генерує графіки у reports/figures
│   ├── train.py                      # GridSearchCV + CV, оцінка, збереження моделі
│   └── predict.py                    # прогноз для одного клієнта (функція + CLI)
├── app/
│   ├── app.py                        # Flask: веб-форма та API
│   └── templates/index.html
├── models/model.joblib, metadata.json  # навчена модель і метадані
├── reports/                          # метрики, таблиці, графіки
├── Dockerfile, requirements.txt, requirements-dev.txt
└── README.md
```

## Аналіз та обробка даних (коротко)
- **Пропуски** `reamining_contract` (~30%) означають «немає контракту» (91% відтоку), тому додано прапорці `no_contract`, `contract_ended`, а значення заповнено 0. Пропуски `download_avg`/`upload_avg` (381 рядок) — прапорець `no_usage_data` + медіана.
- **Викиди й скошеність**: `log1p` для `bill_avg`, `download_avg`, `upload_avg`; від'ємний `subscription_age` обрізається до 0.
- **Стандартизація**: `StandardScaler`. Усі ознаки числові, тому кодування категорій не потрібне.
- **`id` виключено** з ознак: він корелює з `churn` (−0.45) лише через впорядкування даних (витік).
- Імпутер і скейлер лежать у `Pipeline`, тож під час крос-валідації не підглядають у валідаційні дані.

## Модель
Порівнюються Logistic Regression, Random Forest і HistGradientBoosting; для кожної — `GridSearchCV` (5-fold, метрика F1). Найкращі параметри бустингу: `learning_rate=0.1`, `max_depth=None`, `max_iter=200`. Поріг класифікації — 0.5 (`models/metadata.json`).

## Запуск

### 1. Через Docker (рекомендовано)
```bash
docker build -t churn-app .
docker run --rm -p 8000:8000 churn-app
```
Відкрийте http://localhost:8000.

### 2. Локально
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m src.eda      # (опційно) графіки EDA -> reports/figures
python -m src.train    # (опційно, ≈3 хв) перенавчання -> models/model.joblib
python -m app.app      # інтерфейс на http://localhost:8000
```
Усі команди виконуються з кореня репозиторію. Версії бібліотек зафіксовано в `requirements.txt` — модель збережено з `scikit-learn==1.8.0`, і для коректного завантаження версія має збігатися.

## Приклад використання

**Веб-форма:** заповніть поля та натисніть «Спрогнозувати» — з'явиться висновок і шкала ймовірності. Поля `Залишок контракту`, `download_avg`, `upload_avg` можна залишити порожніми (немає контракту / немає даних).

**API:**
```bash
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{
  "is_tv_subscriber": 0, "is_movie_package_subscriber": 0, "subscription_age": 2.5,
  "bill_avg": 20, "reamining_contract": null, "service_failure_count": 0,
  "download_avg": 5, "upload_avg": 1, "download_over_limit": 0}'
```
Відповідь:
```json
{"message": "Клієнт має високу ймовірність відтоку", "probability": 0.9803, "risk": "high"}
```

**Python / CLI:**
```bash
python -m src.predict '{"is_tv_subscriber":1,"is_movie_package_subscriber":1,"subscription_age":3,"bill_avg":20,"reamining_contract":1.5,"service_failure_count":0,"download_avg":60,"upload_avg":4,"download_over_limit":0}'
```

## Обмеження
- Оцінка зроблена на випадковому split; часової компоненти в даних немає.
- Параметри бустингу лежать на межі сітки — розширення сітки могло б дати невеликий приріст.
- Назва колонки `reamining_contract` збережена з оригінального датасету (з помилкою).
