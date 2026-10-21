"""EDA: зберігає графіки у reports/figures та друкує ключові цифри."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import ROOT, TARGET, load_data

FIG = ROOT / "reports" / "figures"


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_data()
    sns.set_theme(style="whitegrid")

    print(df.shape, "дублікатів:", df.duplicated().sum())
    print(df.isna().sum()[lambda s: s > 0])

    plt.figure(figsize=(4, 3.5))
    sns.countplot(x=TARGET, data=df)
    plt.title("Розподіл цільової змінної churn")
    save("01_target_distribution.png")

    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0]
    plt.figure(figsize=(6, 3))
    miss.plot.barh()
    plt.xlabel("% пропусків")
    plt.title("Пропущені значення")
    save("02_missing_values.png")

    num = ["subscription_age", "bill_avg", "reamining_contract", "service_failure_count",
           "download_avg", "upload_avg", "download_over_limit"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), num):
        sns.histplot(df[c].dropna(), bins=40, ax=ax)
        ax.set_title(c)
    axes.ravel()[-1].axis("off")
    save("03_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, c in zip(axes, ["bill_avg", "download_avg", "upload_avg"]):
        sns.boxplot(x=df[c], ax=ax)
        ax.set_title(f"{c} (max={df[c].max():.0f})")
    save("04_outliers.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Кореляційна матриця (включно з id)")
    save("05_correlation.png")

    d = df.copy()
    d["contract_status"] = pd.cut(
        d["reamining_contract"].fillna(-1), [-2, -0.5, 0, 0.5, 1, 3],
        labels=["немає контракту", "закінчився (0)", "0–0.5 р.", "0.5–1 р.", ">1 р."])
    plt.figure(figsize=(7, 3.8))
    sns.barplot(x="contract_status", y=TARGET, data=d, errorbar=None)
    plt.ylabel("частка відтоку")
    plt.xlabel("")
    plt.title("Відтік залежно від залишку контракту")
    save("06_churn_by_contract.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    sns.barplot(x="is_tv_subscriber", y=TARGET, data=df, errorbar=None, ax=axes[0])
    sns.barplot(x="is_movie_package_subscriber", y=TARGET, data=df, errorbar=None, ax=axes[1])
    sns.barplot(x=df["service_failure_count"].clip(upper=5), y=df[TARGET], errorbar=None, ax=axes[2])
    axes[2].set_xlabel("service_failure_count (5 = 5+)")
    for ax in axes:
        ax.set_ylabel("частка відтоку")
    save("07_churn_by_services.png")

    q = pd.qcut(df["download_avg"], 5, duplicates="drop")
    plt.figure(figsize=(7, 3.6))
    df.groupby(q, observed=True)[TARGET].mean().plot.bar()
    plt.ylabel("частка відтоку")
    plt.title("Відтік за квінтилями download_avg")
    save("08_churn_by_download.png")


if __name__ == "__main__":
    main()
