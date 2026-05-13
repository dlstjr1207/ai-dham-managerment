from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("final_data_20_weather.csv")
HORIZON = 3
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
INFLOW_TARGET = "target_inflow_3h"
DISCHARGE_TARGET = "target_discharge_3h"

DAMS = {
    2018110: "남강댐",
    2002110: "임하댐",
}


INFLOW_FEATURES = [
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_lag_6h",
    "inflow_lag_12h",
    "inflow_mean_3h",
    "inflow_mean_6h",
    "inflow_mean_12h",
    "inflow_mean_24h",
    "rain",
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

DISCHARGE_BASE_FEATURES = [
    "totdcwtrqy",
    "discharge_lag_1h",
    "discharge_lag_3h",
    "discharge_lag_6h",
    "discharge_lag_12h",
    "discharge_mean_3h",
    "discharge_mean_6h",
    "discharge_mean_12h",
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
    "rain_sum_48h",
    "rain_sum_72h",
    "lowlevel",
    "rsvwtqy",
    "rsvwtrt",
]


def models():
    return {
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
    }


def metric_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def add_features(df):
    df = df.sort_values("obsrdt").reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
        df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
    for lag in [1, 3, 6, 12]:
        df[f"discharge_lag_{lag}h"] = df["totdcwtrqy"].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = df["inflowqy"].rolling(window, min_periods=window).mean()
    for window in [3, 6, 12]:
        df[f"discharge_mean_{window}h"] = df["totdcwtrqy"].rolling(window, min_periods=window).mean()
    for window in [3, 6, 12, 24, 48, 72]:
        df[f"rain_sum_{window}h"] = df["rain"].rolling(window, min_periods=window).sum()

    df[INFLOW_TARGET] = df["inflowqy"].shift(-HORIZON)
    df[DISCHARGE_TARGET] = df["totdcwtrqy"].shift(-HORIZON)
    df["inflow_change_3h"] = df[INFLOW_TARGET] - df["inflowqy"]
    df["discharge_change_3h"] = df[DISCHARGE_TARGET] - df["totdcwtrqy"]
    return df


def evaluate(y_true, pred, baseline):
    model_metrics = metric_dict(y_true, pred)
    baseline_metrics = metric_dict(y_true, baseline)
    return {
        "baseline_MAE": baseline_metrics["MAE"],
        "baseline_RMSE": baseline_metrics["RMSE"],
        "baseline_R2": baseline_metrics["R2"],
        **model_metrics,
    }


def train_inflow(dam_name, df):
    needed = ["obsrdt", "inflowqy", INFLOW_TARGET] + INFLOW_FEATURES
    data = df[list(dict.fromkeys(needed))].dropna().copy()
    train = data[data["obsrdt"] < TRAIN_END]
    test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)]

    rows = []
    pred_df = test[["obsrdt", "inflowqy", INFLOW_TARGET]].copy()

    for model_name, model in models().items():
        model.fit(train[INFLOW_FEATURES], train[INFLOW_TARGET])
        pred = model.predict(test[INFLOW_FEATURES])
        pred_df[f"pred_inflow_{model_name}"] = pred
        rows.append(
            {
                "dam_name": dam_name,
                "target": "유입량",
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "target_mean": test[INFLOW_TARGET].mean(),
                "target_max": test[INFLOW_TARGET].max(),
                **evaluate(test[INFLOW_TARGET], pred, test["inflowqy"].to_numpy()),
            }
        )

    return pd.DataFrame(rows), pred_df


def train_discharge(dam_name, df, inflow_pred_df, best_inflow_model):
    df = df.merge(
        inflow_pred_df[["obsrdt", f"pred_inflow_{best_inflow_model}"]],
        on="obsrdt",
        how="left",
    )

    feature_sets = {
        "기본피처": DISCHARGE_BASE_FEATURES,
        "실제미래유입량참고": DISCHARGE_BASE_FEATURES + [INFLOW_TARGET],
    }

    rows = []
    pred_frames = []
    for feature_set_name, features in feature_sets.items():
        data = df[list(dict.fromkeys(["obsrdt", "totdcwtrqy", INFLOW_TARGET, DISCHARGE_TARGET] + features))].dropna().copy()
        train = data[data["obsrdt"] < TRAIN_END].copy()
        test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)].copy()

        for model_name, model in models().items():
            model.fit(train[features], train[DISCHARGE_TARGET])
            pred = model.predict(test[features])
            rows.append(
                {
                    "dam_name": dam_name,
                    "target": "방류량",
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "target_mean": test[DISCHARGE_TARGET].mean(),
                    "target_max": test[DISCHARGE_TARGET].max(),
                    **evaluate(test[DISCHARGE_TARGET], pred, test["totdcwtrqy"].to_numpy()),
                }
            )
            pred_out = test[["obsrdt", "totdcwtrqy", DISCHARGE_TARGET]].copy()
            pred_out["prediction"] = pred
            pred_out["feature_set"] = feature_set_name
            pred_out["model"] = model_name
            pred_out["dam_name"] = dam_name
            pred_frames.append(pred_out)

    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True)


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["obsrdt"])
    all_inflow_results = []
    all_discharge_results = []
    all_inflow_predictions = []
    all_discharge_predictions = []

    for dam_code, dam_name in DAMS.items():
        dam = raw[raw["dam_code"] == dam_code].copy()
        dam = add_features(dam)

        inflow_results, inflow_pred = train_inflow(dam_name, dam)
        best_inflow_model = inflow_results.sort_values(["MAE", "RMSE"]).iloc[0]["model"]

        discharge_results, discharge_pred = train_discharge(dam_name, dam, inflow_pred, best_inflow_model)

        inflow_pred["dam_name"] = dam_name
        inflow_pred["dam_code"] = dam_code
        discharge_pred["dam_code"] = dam_code

        all_inflow_results.append(inflow_results)
        all_discharge_results.append(discharge_results)
        all_inflow_predictions.append(inflow_pred)
        all_discharge_predictions.append(discharge_pred)

    inflow_result_df = pd.concat(all_inflow_results, ignore_index=True)
    discharge_result_df = pd.concat(all_discharge_results, ignore_index=True)
    inflow_pred_df = pd.concat(all_inflow_predictions, ignore_index=True)
    discharge_pred_df = pd.concat(all_discharge_predictions, ignore_index=True)

    inflow_result_df.to_csv("남강댐_임하댐_유입량예측_평가결과.csv", index=False, encoding="utf-8-sig")
    discharge_result_df.to_csv("남강댐_임하댐_방류량예측_평가결과.csv", index=False, encoding="utf-8-sig")
    inflow_pred_df.to_csv("남강댐_임하댐_유입량예측_예측결과.csv", index=False, encoding="utf-8-sig")
    discharge_pred_df.to_csv("남강댐_임하댐_방류량예측_예측결과.csv", index=False, encoding="utf-8-sig")

    print("INFLOW")
    print(inflow_result_df.sort_values(["dam_name", "MAE"]).to_string(index=False))
    print("\nDISCHARGE")
    print(discharge_result_df.sort_values(["dam_name", "MAE"]).to_string(index=False))


if __name__ == "__main__":
    main()
