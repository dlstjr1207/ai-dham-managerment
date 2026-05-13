from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("보령댐_운영_강수량_병합데이터.csv")
TARGET = "target_inflow_3h"

FEATURES = [
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_lag_6h",
    "inflow_mean_3h",
    "inflow_mean_6h",
    "inflow_mean_12h",
    "rain_mm",
    "rain_lag_1h",
    "rain_lag_3h",
    "rain_lag_6h",
    "rain_lag_12h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_12h",
    "rain_sum_24h",
    "rain_sum_48h",
    "rain_sum_72h",
    "lowlevel",
    "rsvwtqy",
    "rsvwtrt",
]


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["obsrdt", "rain_base_time"])
    df = df.sort_values("obsrdt").reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = df["inflowqy"].rolling(window, min_periods=window).mean()

    df[TARGET] = df["inflowqy"].shift(-3)
    df["month"] = df["obsrdt"].dt.month
    df["year"] = df["obsrdt"].dt.year
    df["is_rain"] = df["rain_sum_24h"] > 0
    df["is_heavy_rain"] = df["rain_sum_24h"] >= 10
    df["is_monsoon"] = df["month"].between(6, 9)
    df["is_rain_or_monsoon"] = df["is_rain"] | df["is_monsoon"]
    return df


def train_eval(df, subset_name, mask):
    needed = ["obsrdt", TARGET, "inflowqy"] + FEATURES
    needed = list(dict.fromkeys(needed))
    data = df.loc[mask, needed].dropna().copy()

    train = data[(data["obsrdt"] >= "2023-01-01") & (data["obsrdt"] < "2025-01-01")]
    test = data[(data["obsrdt"] >= "2025-01-01") & (data["obsrdt"] < "2026-01-01")]

    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])

    baseline_pred = test["inflowqy"].to_numpy()
    result = {
        "subset": subset_name,
        "rows_after_filter": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train["obsrdt"].min(),
        "train_end": train["obsrdt"].max(),
        "test_start": test["obsrdt"].min(),
        "test_end": test["obsrdt"].max(),
        "target_mean": test[TARGET].mean(),
        "target_max": test[TARGET].max(),
        "baseline_MAE": metrics(test[TARGET], baseline_pred)["MAE"],
        "baseline_RMSE": metrics(test[TARGET], baseline_pred)["RMSE"],
        "baseline_R2": metrics(test[TARGET], baseline_pred)["R2"],
        **metrics(test[TARGET], pred),
    }

    pred_df = test[["obsrdt", TARGET, "inflowqy"]].copy()
    pred_df["prediction"] = pred
    pred_df["error"] = pred_df["prediction"] - pred_df[TARGET]
    pred_df["subset"] = subset_name

    importance_df = pd.DataFrame(
        {
            "subset": subset_name,
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return result, pred_df, importance_df


def main():
    df = prepare_data()

    subsets = {
        "강우일_24시간누적강수_0초과": df["is_rain"],
        "강한강우_24시간누적강수_10mm이상": df["is_heavy_rain"],
        "장마철_6월부터9월": df["is_monsoon"],
        "강우일또는장마철": df["is_rain_or_monsoon"],
    }

    results = []
    predictions = []
    importances = []

    for name, mask in subsets.items():
        result, pred_df, importance_df = train_eval(df, name, mask)
        results.append(result)
        predictions.append(pred_df)
        importances.append(importance_df)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True)
    importances_df = pd.concat(importances, ignore_index=True)

    results_df.to_csv("보령댐_강우장마철_유입량예측_평가결과.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv("보령댐_강우장마철_유입량예측_예측결과.csv", index=False, encoding="utf-8-sig")
    importances_df.to_csv("보령댐_강우장마철_유입량예측_피처중요도.csv", index=False, encoding="utf-8-sig")

    print(results_df.to_string(index=False))
    print("\nTOP FEATURES")
    for subset in results_df["subset"]:
        top = importances_df[importances_df["subset"] == subset].head(8)
        print(f"\n{subset}")
        print(top[["feature", "importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
