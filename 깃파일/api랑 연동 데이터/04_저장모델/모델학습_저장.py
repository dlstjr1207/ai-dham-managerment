from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_PATH = PROJECT_ROOT / "final_data_20_weather.csv"
MASTER_PATH = BASE_DIR.parent / "00_DB설정" / "20개댐_마스터.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

HORIZON = 3
INFLOW_TARGET = "target_inflow_3h"
DISCHARGE_TARGET = "target_discharge_3h"
CHANGE_TARGET = "target_discharge_change"

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

DISCHARGE_FEATURES = [
    "totdcwtrqy",
    "discharge_lag_1h",
    "discharge_lag_3h",
    "discharge_lag_6h",
    "discharge_lag_12h",
    "discharge_mean_3h",
    "discharge_mean_6h",
    "discharge_mean_12h",
    "discharge_mean_24h",
    "discharge_change_lag_1h",
    "discharge_change_lag_3h",
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_lag_6h",
    "inflow_lag_12h",
    "inflow_mean_3h",
    "inflow_mean_6h",
    "inflow_mean_12h",
    "inflow_mean_24h",
    "predicted_inflow_3h",
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("obsrdt").reset_index(drop=True)
    df["rain"] = df["rain"].fillna(df["rf"]).fillna(0)
    df["tmp"] = df["tmp"].fillna(0)
    df["snow"] = df["snow"].fillna(0)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
        df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
        df[f"discharge_lag_{lag}h"] = df["totdcwtrqy"].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = df["inflowqy"].rolling(window, min_periods=window).mean()
        df[f"discharge_mean_{window}h"] = df["totdcwtrqy"].rolling(window, min_periods=window).mean()

    for window in [3, 6, 12, 24, 48, 72]:
        df[f"rain_sum_{window}h"] = df["rain"].rolling(window, min_periods=window).sum()

    df["discharge_change_lag_1h"] = df["totdcwtrqy"].diff(1)
    df["discharge_change_lag_3h"] = df["totdcwtrqy"] - df["totdcwtrqy"].shift(3)
    df[INFLOW_TARGET] = df["inflowqy"].shift(-HORIZON)
    df[DISCHARGE_TARGET] = df["totdcwtrqy"].shift(-HORIZON)
    df[CHANGE_TARGET] = (df[DISCHARGE_TARGET] - df["totdcwtrqy"]).abs() >= 5
    return df


def train_one_dam(raw: pd.DataFrame, dam_code: int, dam_name: str) -> dict:
    dam = add_features(raw[raw["dam_code"] == dam_code].copy())

    inflow_data = dam[["obsrdt", INFLOW_TARGET] + INFLOW_FEATURES].dropna().copy()
    inflow_model = GradientBoostingRegressor(random_state=42)
    inflow_model.fit(inflow_data[INFLOW_FEATURES], inflow_data[INFLOW_TARGET])
    dam["predicted_inflow_3h"] = pd.NA
    inflow_feature_rows = dam[INFLOW_FEATURES].notna().all(axis=1)
    dam.loc[inflow_feature_rows, "predicted_inflow_3h"] = inflow_model.predict(dam.loc[inflow_feature_rows, INFLOW_FEATURES])

    discharge_needed = [DISCHARGE_TARGET, CHANGE_TARGET] + DISCHARGE_FEATURES
    discharge_data = dam[discharge_needed].dropna().copy()

    discharge_model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    discharge_model.fit(discharge_data[DISCHARGE_FEATURES], discharge_data[DISCHARGE_TARGET])

    change_model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    change_model.fit(discharge_data[DISCHARGE_FEATURES], discharge_data[CHANGE_TARGET].astype(int))

    inflow_pred = inflow_model.predict(inflow_data[INFLOW_FEATURES])
    discharge_pred = discharge_model.predict(discharge_data[DISCHARGE_FEATURES])
    change_pred = change_model.predict(discharge_data[DISCHARGE_FEATURES])
    change_prob = change_model.predict_proba(discharge_data[DISCHARGE_FEATURES])[:, 1]

    auc = None
    if discharge_data[CHANGE_TARGET].nunique() > 1:
        auc = roc_auc_score(discharge_data[CHANGE_TARGET].astype(int), change_prob)

    bundle = {
        "dam_code": dam_code,
        "dam_name": dam_name,
        "horizon_hours": HORIZON,
        "inflow_features": INFLOW_FEATURES,
        "discharge_features": DISCHARGE_FEATURES,
        "inflow_model": inflow_model,
        "discharge_model": discharge_model,
        "change_model": change_model,
        "model_version": "ml-v0.1",
    }
    model_path = MODEL_DIR / f"dam_{dam_code}_models.joblib"
    joblib.dump(bundle, model_path)

    return {
        "dam_code": dam_code,
        "dam_name": dam_name,
        "model_path": str(model_path),
        "train_rows_inflow": len(inflow_data),
        "train_rows_discharge": len(discharge_data),
        "inflow_MAE": mean_absolute_error(inflow_data[INFLOW_TARGET], inflow_pred),
        "inflow_R2": r2_score(inflow_data[INFLOW_TARGET], inflow_pred),
        "discharge_MAE": mean_absolute_error(discharge_data[DISCHARGE_TARGET], discharge_pred),
        "discharge_R2": r2_score(discharge_data[DISCHARGE_TARGET], discharge_pred),
        "change_accuracy": accuracy_score(discharge_data[CHANGE_TARGET].astype(int), change_pred),
        "change_auc": auc,
        "change_rate": discharge_data[CHANGE_TARGET].mean(),
    }


def main() -> None:
    raw = pd.read_csv(DATA_PATH, parse_dates=["obsrdt"])
    master = pd.read_csv(MASTER_PATH)

    rows = []
    for dam in master.itertuples(index=False):
        print(f"{dam.dam_name} 모델 학습/저장 중...", flush=True)
        rows.append(train_one_dam(raw, int(dam.dam_code), dam.dam_name))

    summary = pd.DataFrame(rows)
    summary.to_csv(MODEL_DIR / "모델학습_요약.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model_version": "ml-v0.1",
        "horizon_hours": HORIZON,
        "inflow_features": INFLOW_FEATURES,
        "discharge_features": DISCHARGE_FEATURES,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("모델 저장 완료")
    print(summary[["dam_name", "inflow_MAE", "discharge_MAE", "change_accuracy", "change_auc", "change_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
