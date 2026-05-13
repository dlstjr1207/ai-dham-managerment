from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("보령댐_운영_강수량_병합데이터.csv")
TARGET_TEMPLATE = "target_totdcwtrqy_{horizon}h"
TEST_START = pd.Timestamp("2026-01-01 00:00:00")

OPERATION_FEATURES = [
    "inflowqy",
    "lowlevel",
    "rf",
    "rsvwtqy",
    "rsvwtrt",
]

CURRENT_DISCHARGE_FEATURE = ["totdcwtrqy"]

RAIN_FEATURES = [
    "rain_mm",
    "rain_lag_1h",
    "rain_lag_2h",
    "rain_lag_3h",
    "rain_lag_6h",
    "rain_lag_12h",
    "rain_lag_24h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_12h",
    "rain_sum_24h",
    "rain_sum_48h",
    "rain_sum_72h",
]


def make_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate(df, horizon, feature_set_name, features):
    target = TARGET_TEMPLATE.format(horizon=horizon)
    needed = ["obsrdt", "totdcwtrqy", target] + features
    needed = list(dict.fromkeys(needed))
    clean = df[needed].dropna().copy()

    train = clean[clean["obsrdt"] < TEST_START]
    test = clean[clean["obsrdt"] >= TEST_START]

    model = XGBRegressor(objective="reg:squarederror")
    model.fit(train[features], train[target])

    pred = model.predict(test[features])
    metrics = make_metrics(test[target], pred)

    train_mean_pred = np.full(len(test), train[target].mean())
    current_discharge_pred = test["totdcwtrqy"].to_numpy()
    train_mean_metrics = make_metrics(test[target], train_mean_pred)
    current_discharge_metrics = make_metrics(test[target], current_discharge_pred)

    pred_df = test[["obsrdt", target]].copy()
    pred_df["prediction"] = pred
    pred_df["error"] = pred_df["prediction"] - pred_df[target]
    pred_df["horizon_hours"] = horizon
    pred_df["feature_set"] = feature_set_name

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
            "horizon_hours": horizon,
            "feature_set": feature_set_name,
        }
    ).sort_values("importance", ascending=False)

    result = {
        "horizon_hours": horizon,
        "feature_set": feature_set_name,
        "features": features,
        "raw_rows": len(df),
        "clean_rows": len(clean),
        "removed_rows": len(df) - len(clean),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train["obsrdt"].min(),
        "train_end": train["obsrdt"].max(),
        "test_start": test["obsrdt"].min(),
        "test_end": test["obsrdt"].max(),
        "test_target_mean": test[target].mean(),
        "test_target_std": test[target].std(),
        "train_mean_baseline_MAE": train_mean_metrics["MAE"],
        "current_discharge_baseline_MAE": current_discharge_metrics["MAE"],
        "current_discharge_baseline_RMSE": current_discharge_metrics["RMSE"],
        "current_discharge_baseline_R2": current_discharge_metrics["R2"],
        **metrics,
    }
    return result, pred_df, importance


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["obsrdt", "rain_base_time"])
    df = df.sort_values("obsrdt").reset_index(drop=True)

    # 현재 시각의 실제 관측값만 입력으로 사용하고, 목표값은 미래 총방류량으로 만든다.
    for horizon in [1, 3, 6]:
        df[TARGET_TEMPLATE.format(horizon=horizon)] = df["totdcwtrqy"].shift(-horizon)

    experiments = [
        ("operation_only", OPERATION_FEATURES),
        ("rain_only", RAIN_FEATURES),
        ("operation_plus_rain", OPERATION_FEATURES + RAIN_FEATURES),
        (
            "operation_plus_current_discharge",
            OPERATION_FEATURES + CURRENT_DISCHARGE_FEATURE,
        ),
        (
            "operation_rain_plus_current_discharge",
            OPERATION_FEATURES + RAIN_FEATURES + CURRENT_DISCHARGE_FEATURE,
        ),
    ]

    results = []
    predictions = []
    importances = []

    for horizon in [1, 3, 6]:
        for name, features in experiments:
            result, pred_df, importance = train_and_evaluate(df, horizon, name, features)
            results.append(result)
            predictions.append(pred_df)
            importances.append(importance)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True)
    importances_df = pd.concat(importances, ignore_index=True)

    results_df.to_csv("보령댐_XGBoost_1시간3시간6시간_평가결과.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv("보령댐_XGBoost_1시간3시간6시간_예측결과.csv", index=False, encoding="utf-8-sig")
    importances_df.to_csv("보령댐_XGBoost_1시간3시간6시간_피처중요도.csv", index=False, encoding="utf-8-sig")

    print("RESULTS")
    print(results_df.to_string(index=False))
    print("\nTOP_IMPORTANCE")
    for horizon in sorted(results_df["horizon_hours"].unique()):
        for name in results_df["feature_set"].unique():
            top = importances_df[
                (importances_df["horizon_hours"] == horizon)
                & (importances_df["feature_set"] == name)
            ].head(8)
            print(f"\n{horizon}h / {name}")
            print(top[["feature", "importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
