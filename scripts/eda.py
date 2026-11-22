# 用途：探索 Food.com 食譜推薦資料的互動長尾、年度變化與評估樣本篩選。
# 流程：由 FOODCOM_DATA_DIR 讀取並清理原始資料，再將三張圖與 CSV 寫入 eda/。

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["FOODCOM_DATA_DIR"])
OUT = ROOT / "eda"
PERIODS = (
    ("2013-01-01", "2016-01-01"),
    ("2016-01-01", "2019-01-01"),
    ("2018-01-01", "2019-01-01"),
)


# 與推薦實驗使用相同的時間有效性規則，避免將尚未發表的食譜評分算進歷史。
def load_data():
    recipes = pd.read_csv(DATA / "RAW_recipes.csv", usecols=["id", "submitted"])
    events = pd.read_csv(
        DATA / "RAW_interactions.csv",
        usecols=["user_id", "recipe_id", "date", "rating"],
    )
    quality = {
        "raw_recipes": len(recipes),
        "raw_interactions": len(events),
        "missing_used_fields": int(recipes.isna().sum().sum() + events.isna().sum().sum()),
        "duplicate_recipe_ids": int(recipes.id.duplicated().sum()),
        "duplicate_user_recipe_pairs": int(events.duplicated(["user_id", "recipe_id"]).sum()),
        "unknown_recipe_ids": int((~events.recipe_id.isin(recipes.id)).sum()),
        "invalid_ratings": int((~events.rating.between(0, 5)).sum()),
        "invalid_dates": int(
            pd.to_datetime(recipes.submitted, format="%Y-%m-%d", errors="coerce").isna().sum()
            + pd.to_datetime(events.date, format="%Y-%m-%d", errors="coerce").isna().sum()
        ),
    }
    if any(quality[key] for key in (
        "missing_used_fields", "duplicate_recipe_ids", "duplicate_user_recipe_pairs",
        "unknown_recipe_ids", "invalid_ratings", "invalid_dates",
    )):
        raise ValueError(f"Raw data failed quality checks: {quality}")
    submitted = recipes.set_index("id").submitted
    valid = events.date.ge(events.recipe_id.map(submitted))
    quality["reviews_before_recipe_submission"] = int((~valid).sum())
    quality["clean_interactions"] = int(valid.sum())
    return recipes, events.loc[valid].copy(), quality


# 以 2016 年預測日的訓練資料觀察稀疏程度，和主要測試設定保持一致。
def interaction_tail(recipes, events):
    origin = "2016-01-01"
    catalog = recipes[recipes.submitted.lt(origin)]
    past = events[events.date.lt(origin)]
    likes = past[past.rating.ge(4)]
    user_counts = likes.user_id.value_counts()
    recipe_counts = likes.recipe_id.value_counts().reindex(catalog.id, fill_value=0)

    bins = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 3000, 10000, 100000])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, counts, title in (
        (axes[0], user_counts, "Positive ratings per user"),
        (axes[1], recipe_counts[recipe_counts.gt(0)], "Recipes with at least one positive rating"),
    ):
        ax.hist(counts, bins=bins, color="#3A6B86", edgecolor="white")
        ax.set(xscale="log", yscale="log", xlabel="Number of 4–5 star ratings", ylabel="Count", title=title)
        ax.grid(alpha=0.2)
    fig.suptitle("Training history before 2016-01-01")
    fig.savefig(OUT / "interaction_tail.png", dpi=160)
    plt.close(fig)

    top_count = max(1, int(np.ceil(len(catalog) * 0.01)))
    return pd.DataFrame(
        [
            ("2016_catalog_recipes", len(catalog)),
            ("2016_prior_positive_ratings", len(likes)),
            ("2016_users_with_prior_likes", len(user_counts)),
            ("2016_recipes_without_prior_likes", int(recipe_counts.eq(0).sum())),
            ("2016_recipes_with_at_most_one_prior_like", int(recipe_counts.le(1).sum())),
            ("2016_median_likes_per_active_user", float(user_counts.median())),
            ("2016_median_likes_per_recipe", float(recipe_counts.median())),
            ("2016_top_1pct_recipes_share_of_likes", float(recipe_counts.nlargest(top_count).sum() / len(likes))),
        ],
        columns=["metric", "value"],
    )


# 以年度統計資料供給量與評分行為，協助判讀不同預測日的資料差異。
def annual_trends(recipes, events):
    cards = recipes.groupby(recipes.submitted.str[:4]).size().rename("new_recipes")
    reviews = events.assign(year=events.date.str[:4]).groupby("year").agg(
        interactions=("rating", "size"),
        active_users=("user_id", "nunique"),
        positive_ratings=("rating", lambda x: int(x.ge(4).sum())),
        zero_ratings=("rating", lambda x: int(x.eq(0).sum())),
    )
    annual = reviews.join(cards, how="outer").fillna(0).astype(int).sort_index()
    annual.index.name = "year"
    annual["positive_rate"] = annual.positive_ratings / annual.interactions.replace(0, np.nan)
    annual["positive_rate_nonzero"] = annual.positive_ratings / (
        annual.interactions - annual.zero_ratings
    ).replace(0, np.nan)
    annual["zero_rate"] = annual.zero_ratings / annual.interactions.replace(0, np.nan)
    annual.to_csv(OUT / "annual.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    years = annual.index.astype(int)
    axes[0].bar(years, annual.new_recipes, color="#3A6B86")
    axes[0].set(ylabel="New recipes", title="Food.com dataset records by year")
    axes[1].bar(years, annual.interactions, color="#6BA687", label="All ratings")
    axes[1].plot(years, annual.positive_ratings, color="#A33E36", marker="o", markersize=3, label="4–5 star ratings")
    axes[1].set(xlabel="Year", ylabel="Interactions")
    axes[1].set_xticks(years[::2])
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "annual_trends.png", dpi=160)
    plt.close(fig)
    return annual


# 逐步重建使用者入選條件，量化評估樣本相對於歷史使用者的篩選比例。
def cohort_selection(recipes, events):
    rows = []
    for origin, end in PERIODS:
        past = events[events.date.lt(origin)]
        counts = past.loc[past.rating.ge(4), "user_id"].value_counts()
        eligible = counts[counts.ge(5)]
        future = events[events.date.ge(origin) & events.date.lt(end) & events.rating.ge(4)]
        future = future[future.user_id.isin(eligible.index)]
        catalog_ids = pd.Index(recipes.loc[recipes.submitted.lt(origin), "id"])
        selected = future[future.recipe_id.isin(catalog_ids)].user_id.unique()
        rows.append(
            {
                "origin": origin,
                "end": end,
                "published_recipes": len(catalog_ids),
                "prior_users": past.user_id.nunique(),
                "prior_likes": int(counts.sum()),
                "users_with_5_likes": len(eligible),
                "eligible_with_future_like": future.user_id.nunique(),
                "evaluated_users": len(selected),
                "evaluated_share_of_eligible": len(selected) / len(eligible),
                "median_prior_likes_evaluated": float(eligible.reindex(selected).median()),
                "median_prior_likes_other_eligible": float(eligible[~eligible.index.isin(selected)].median()),
            }
        )
    cohort = pd.DataFrame(rows)
    cohort.to_csv(OUT / "cohort_selection.csv", index=False)

    cols = ["prior_users", "users_with_5_likes", "eligible_with_future_like", "evaluated_users"]
    labels = ["Prior users", "5+ prior likes", "Future like", "Eligible target"]
    x = np.arange(len(cohort))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for index, (col, label) in enumerate(zip(cols, labels)):
        ax.bar(x + (index - 1.5) * width, cohort[col], width, label=label)
    ax.set(yscale="log", ylabel="Users (log scale)", xticks=x, xticklabels=cohort.origin, title="Evaluation cohort selection")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "cohort_selection.png", dpi=160)
    plt.close(fig)
    return cohort


# 主流程只負責串接三項分析，方便重跑並核對產物。
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recipes, events, quality = load_data()
    pd.Series(quality, name="value").rename_axis("metric").to_csv(OUT / "data_quality.csv")
    summary = interaction_tail(recipes, events)
    summary.to_csv(OUT / "summary.csv", index=False)
    annual_trends(recipes, events)
    cohort = cohort_selection(recipes, events)
    print(f"Wrote EDA tables and figures to {OUT}")
    print(cohort[["origin", "users_with_5_likes", "evaluated_users"]].to_string(index=False))


if __name__ == "__main__":
    main()
