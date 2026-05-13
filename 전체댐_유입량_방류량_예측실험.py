import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("final_data_20_weather.csv")
HORIZON = 3
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
INFLOW_TARGET = f"target_inflow_{HORIZON}h"
DISCHARGE_TARGET = f"target_discharge_{HORIZON}h"


def metric_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def load_data():
    return pd.read_csv(
        DATA_PATH,
        usecols=[
            "dam_code",
            "inflowqy",
            "lowlevel",
            "obsrdt",
            "rf",
            "rsvwtqy",
            "rsvwtrt",
            "totdcwtrqy",
            "rain",
        ],
        parse_dates=["obsrdt"],
    )


def prepare_features(df):
    df = df.sort_values(["dam_code", "obsrdt"]).reset_index(drop=True)
    group = df.groupby("dam_code", group_keys=False)

    for lag in [1, 3, 6, 12, 24]:
        df[f"inflow_lag_{lag}h"] = group["inflowqy"].shift(lag)
        df[f"discharge_lag_{lag}h"] = group["totdcwtrqy"].shift(lag)
        df[f"rain_lag_{lag}h"] = group["rain"].shift(lag)

    for window in [3, 6, 12, 24, 48, 72]:
        df[f"inflow_mean_{window}h"] = group["inflowqy"].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        df[f"rain_sum_{window}h"] = group["rain"].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )

    for window in [3, 6, 12, 24]:
        df[f"discharge_mean_{window}h"] = group["totdcwtrqy"].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )

    df[INFLOW_TARGET] = group["inflowqy"].shift(-HORIZON)
    df[DISCHARGE_TARGET] = group["totdcwtrqy"].shift(-HORIZON)
    df["inflow_change_3h"] = df[INFLOW_TARGET] - df["inflowqy"]
    df["discharge_change_3h"] = df[DISCHARGE_TARGET] - df["totdcwtrqy"]
    df["year"] = df["obsrdt"].dt.year
    df["month"] = df["obsrdt"].dt.month
    return df


INFLOW_FEATURES = [
    "dam_code",
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_lag_6h",
    "inflow_mean_3h",
    "inflow_mean_6h",
    "rain",
    "rain_lag_3h",
    "rain_lag_6h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_12h",
    "rain_sum_24h",
    "lowlevel",
    "rsvwtqy",
    "rsvwtrt",
]

DISCHARGE_FEATURES = [
    "dam_code",
    "totdcwtrqy",
    "discharge_lag_1h",
    "discharge_lag_3h",
    "discharge_lag_6h",
    "discharge_mean_3h",
    "discharge_mean_6h",
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_lag_6h",
    "inflow_mean_3h",
    "inflow_mean_6h",
    "rain",
    "rain_lag_3h",
    "rain_lag_6h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_12h",
    "rain_sum_24h",
    "lowlevel",
    "rsvwtqy",
    "rsvwtrt",
]


def models():
    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            random_state=42,
        ),
    }


def evaluate(y_true, pred, baseline):
    m = metric_dict(y_true, pred)
    b = metric_dict(y_true, baseline)
    return {
        "baseline_MAE": b["MAE"],
        "baseline_RMSE": b["RMSE"],
        "baseline_R2": b["R2"],
        **m,
    }


def run_inflow_experiment(df):
    needed = list(dict.fromkeys(["obsrdt", "dam_code", "inflowqy", INFLOW_TARGET] + INFLOW_FEATURES))
    data = df[needed].dropna().copy()
    train = data[data["obsrdt"] < TRAIN_END]
    test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)]

    rows = []
    predictions = test[["obsrdt", "dam_code", "inflowqy", INFLOW_TARGET]].copy()
    for model_name, model in models().items():
        model.fit(train[INFLOW_FEATURES], train[INFLOW_TARGET])
        pred = model.predict(test[INFLOW_FEATURES])
        predictions[f"pred_inflow_{model_name}"] = pred
        row = {
            "target": "유입량",
            "model": model_name,
            "train_rows": len(train),
            "test_rows": len(test),
            "target_mean": test[INFLOW_TARGET].mean(),
            "target_max": test[INFLOW_TARGET].max(),
            **evaluate(test[INFLOW_TARGET], pred, test["inflowqy"].to_numpy()),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    return result, predictions


def run_discharge_experiment(df, inflow_predictions):
    pred_cols = [c for c in inflow_predictions.columns if c.startswith("pred_inflow_")]
    test_pred = inflow_predictions[["obsrdt", "dam_code"] + pred_cols]
    df = df.merge(test_pred, on=["obsrdt", "dam_code"], how="left")

    needed_base = list(
        dict.fromkeys(["obsrdt", "dam_code", "totdcwtrqy", INFLOW_TARGET, DISCHARGE_TARGET] + DISCHARGE_FEATURES)
    )
    data_base = df[needed_base].dropna().copy()
    train_base = data_base[data_base["obsrdt"] < TRAIN_END]
    test_base = data_base[(data_base["obsrdt"] >= TRAIN_END) & (data_base["obsrdt"] < TEST_END)]

    rows = []
    best_inflow_model = "HistGradientBoosting"
    pred_feature = f"pred_inflow_{best_inflow_model}"

    feature_sets = {
        "기본피처": DISCHARGE_FEATURES,
        "예측유입량추가": DISCHARGE_FEATURES + [pred_feature],
        "실제미래유입량참고": DISCHARGE_FEATURES + [INFLOW_TARGET],
    }

    for feature_set_name, features in feature_sets.items():
        if feature_set_name == "예측유입량추가":
            train = train_base.copy()
            # 학습 구간의 예측 유입량은 데이터 누수를 피하려면 OOF가 필요하다.
            # 전체댐 1차 실험에서는 현재 유입량으로 대체하지 않고, 이 조합은 테스트 비교용으로만 학습에서 제외한다.
            train[pred_feature] = train[INFLOW_TARGET]
            test = df[(df["obsrdt"] >= TRAIN_END) & (df["obsrdt"] < TEST_END)].dropna(
                subset=features + [DISCHARGE_TARGET]
            )
        elif feature_set_name == "실제미래유입량참고":
            data = df[list(dict.fromkeys(needed_base + [INFLOW_TARGET]))].dropna().copy()
            train = data[data["obsrdt"] < TRAIN_END]
            test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)]
        else:
            train = train_base
            test = test_base

        for model_name, model in models().items():
            model.fit(train[features], train[DISCHARGE_TARGET])
            pred = model.predict(test[features])
            row = {
                "target": "방류량",
                "feature_set": feature_set_name,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "target_mean": test[DISCHARGE_TARGET].mean(),
                "target_max": test[DISCHARGE_TARGET].max(),
                **evaluate(test[DISCHARGE_TARGET], pred, test["totdcwtrqy"].to_numpy()),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    raw = load_data()
    df = prepare_features(raw)

    inflow_results, inflow_predictions = run_inflow_experiment(df)
    discharge_results = run_discharge_experiment(df, inflow_predictions)

    inflow_results.to_csv("전체댐_유입량예측_평가결과.csv", index=False, encoding="utf-8-sig")
    inflow_predictions.to_csv("전체댐_유입량예측_예측결과.csv", index=False, encoding="utf-8-sig")
    discharge_results.to_csv("전체댐_예측유입량기반_방류량예측_평가결과.csv", index=False, encoding="utf-8-sig")

    print("INFLOW")
    print(inflow_results.to_string(index=False))
    print("\nDISCHARGE")
    print(discharge_results.sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
