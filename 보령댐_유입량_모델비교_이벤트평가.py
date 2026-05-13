from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DATA_PATH = Path("보령댐_운영_강수량_병합데이터.csv")
TARGET = "target_inflow_3h"
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")

FEATURES = [
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_lag_6h",
    "inflow_lag_12h",
    "inflow_mean_3h",
    "inflow_mean_6h",
    "inflow_mean_12h",
    "inflow_mean_24h",
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


def make_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["obsrdt", "rain_base_time"])
    df = df.sort_values("obsrdt").reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = df["inflowqy"].rolling(window, min_periods=window).mean()

    df[TARGET] = df["inflowqy"].shift(-3)
    df["inflow_change_3h"] = df[TARGET] - df["inflowqy"]
    df["inflow_change_ratio_3h"] = np.where(
        df["inflowqy"].abs() > 1e-9,
        df["inflow_change_3h"] / df["inflowqy"].abs(),
        np.nan,
    )
    df["month"] = df["obsrdt"].dt.month
    df["is_rain"] = df["rain_sum_24h"] > 0
    df["is_heavy_rain"] = df["rain_sum_24h"] >= 10
    df["is_monsoon"] = df["month"].between(6, 9)

    needed = ["obsrdt", TARGET, "inflowqy", "inflow_change_3h", "inflow_change_ratio_3h"] + FEATURES
    needed = list(dict.fromkeys(needed))
    data = df[needed].dropna().copy()

    train = data[data["obsrdt"] < TEST_START].copy()
    test = data[(data["obsrdt"] >= TEST_START) & (data["obsrdt"] < TEST_END)].copy()

    increase_threshold = train["inflow_change_3h"].quantile(0.95)
    high_inflow_threshold = train[TARGET].quantile(0.95)

    data["event_increase_top5"] = data["inflow_change_3h"] >= increase_threshold
    data["event_high_inflow_top5"] = data[TARGET] >= high_inflow_threshold
    data["event_rain_and_increase"] = (data["rain_sum_24h"] >= 10) & data["event_increase_top5"]

    train = data[data["obsrdt"] < TEST_START].copy()
    test = data[(data["obsrdt"] >= TEST_START) & (data["obsrdt"] < TEST_END)].copy()
    return data, train, test, increase_threshold, high_inflow_threshold


def models():
    return {
        "Ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
        "MLP": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        max_iter=1200,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def metric_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def eval_predictions(name, train_scope, eval_scope, model_name, y_true, pred, baseline):
    model_metrics = metric_dict(y_true, pred)
    baseline_metrics = metric_dict(y_true, baseline)
    return {
        "train_scope": train_scope,
        "eval_scope": eval_scope,
        "model": model_name,
        "eval_rows": len(y_true),
        "target_mean": y_true.mean(),
        "target_max": y_true.max(),
        "baseline_MAE": baseline_metrics["MAE"],
        "baseline_RMSE": baseline_metrics["RMSE"],
        "baseline_R2": baseline_metrics["R2"],
        **model_metrics,
    }


def main():
    _, train, test, increase_threshold, high_inflow_threshold = make_data()

    eval_masks = {
        "2025_전체": pd.Series(True, index=test.index),
        "2025_강우일": test["rain_sum_24h"] > 0,
        "2025_강한강우": test["rain_sum_24h"] >= 10,
        "2025_장마철": test["obsrdt"].dt.month.between(6, 9),
        "2025_유입량증가상위5퍼센트": test["inflow_change_3h"] >= increase_threshold,
        "2025_유입량상위5퍼센트": test[TARGET] >= high_inflow_threshold,
        "2025_강한강우_그리고_유입량증가상위5퍼센트": (test["rain_sum_24h"] >= 10)
        & (test["inflow_change_3h"] >= increase_threshold),
    }

    train_scopes = {
        "2023_2024_전체학습": train,
        "2023_2024_강우장마철학습": train[
            (train["rain_sum_24h"] > 0) | train["obsrdt"].dt.month.between(6, 9)
        ],
        "2023_2024_유입량증가상위5퍼센트학습": train[
            train["inflow_change_3h"] >= increase_threshold
        ],
    }

    rows = []
    prediction_rows = []

    for train_scope_name, train_df in train_scopes.items():
        for model_name, model in models().items():
            model.fit(train_df[FEATURES], train_df[TARGET])

            for eval_scope_name, mask in eval_masks.items():
                eval_df = test.loc[mask].copy()
                if len(eval_df) < 20:
                    continue

                pred = model.predict(eval_df[FEATURES])
                baseline = eval_df["inflowqy"].to_numpy()
                rows.append(
                    eval_predictions(
                        "inflow",
                        train_scope_name,
                        eval_scope_name,
                        model_name,
                        eval_df[TARGET],
                        pred,
                        baseline,
                    )
                )

                if eval_scope_name in [
                    "2025_유입량증가상위5퍼센트",
                    "2025_유입량상위5퍼센트",
                ]:
                    pred_df = eval_df[["obsrdt", "inflowqy", TARGET, "inflow_change_3h"]].copy()
                    pred_df["prediction"] = pred
                    pred_df["error"] = pred_df["prediction"] - pred_df[TARGET]
                    pred_df["train_scope"] = train_scope_name
                    pred_df["eval_scope"] = eval_scope_name
                    pred_df["model"] = model_name
                    prediction_rows.append(pred_df)

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    results.to_csv("보령댐_유입량_여러모델_이벤트평가결과.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv("보령댐_유입량_여러모델_이벤트예측결과.csv", index=False, encoding="utf-8-sig")

    print("threshold_inflow_change_top5", increase_threshold)
    print("threshold_high_inflow_top5", high_inflow_threshold)
    print("\nBEST_BY_SCOPE")
    best = results.sort_values("MAE").groupby(["train_scope", "eval_scope"], as_index=False).first()
    print(
        best[
            [
                "train_scope",
                "eval_scope",
                "model",
                "eval_rows",
                "baseline_MAE",
                "MAE",
                "baseline_R2",
                "R2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
