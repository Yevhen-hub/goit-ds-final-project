"""Попередня обробка даних: завантаження, інженерія ознак і sklearn-пайплайн."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "internet_service_churn.csv"
TARGET = "churn"

# Сирі ознаки, які вводить користувач (id не використовується: це лише ідентифікатор)
RAW_FEATURES = [
    "is_tv_subscriber",
    "is_movie_package_subscriber",
    "subscription_age",
    "bill_avg",
    "reamining_contract",  # назва з помилкою збережена як в оригінальному датасеті
    "service_failure_count",
    "download_avg",
    "upload_avg",
    "download_over_limit",
]
SKEWED = ["bill_avg", "download_avg", "upload_avg"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[RAW_FEATURES], df[TARGET]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Детерміновані перетворення (не залежать від даних, тому не дають витоку).

    - no_contract: пропуск у reamining_contract = у клієнта немає контракту
      (91% таких клієнтів пішли), тому пропуск є інформацією, а не шумом;
    - contract_ended: контракт є, але залишилось 0 років (99.6% відтоку);
    - no_usage_data: немає даних про трафік (усі 381 такі клієнти не пішли);
    - subscription_age < 0 -> 0 (некоректне значення);
    - log1p для сильно скошених bill/download/upload (викиди до 4415 ГБ).
    """
    X = X.copy()
    X["no_contract"] = X["reamining_contract"].isna().astype(int)
    X["contract_ended"] = (X["reamining_contract"] == 0).astype(int)
    X["no_usage_data"] = X["download_avg"].isna().astype(int)
    X["reamining_contract"] = X["reamining_contract"].fillna(0)
    X["subscription_age"] = X["subscription_age"].clip(lower=0)
    for col in SKEWED:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X


def build_pipeline(model) -> Pipeline:
    """features -> заповнення пропусків (медіана) -> стандартизація -> модель.

    Медіана та середнє/std рахуються лише на train-фолдах (усередині CV).
    """
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
