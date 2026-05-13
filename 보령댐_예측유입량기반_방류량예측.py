from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


DATA_PATH = Path("보령댐_운영_강수량_병합데이터.csv")
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
INFLOW_TARGET = "target_inflow_3h"
DISCHARGE_TARGET = "target_discharge_3h"


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

BASE_DISCHARGE_FEATURES = [
    "totdcwtrqy",
    "discharge_lag_1h",
    "discharge_lag_3h",
    "discharge_lag_6h",
    "discharge_mean_3h",
    "discharge_mean_6h",
    "inflowqy",
    "inflow_lag_1h",
    "inflow_lag_3h",
    "inflow_mean_3h",
    "rain_mm",
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


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def make_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["obsrdt", "rain_base_time"])
    df = df.sort_values("obsrdt").reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = df["inflowqy"].rolling(window, min_periods=window).mean()

    for lag in [1, 3, 6]:
        df[f"discharge_lag_{lag}h"] = df["totdcwtrqy"].shift(lag)
    for window in [3, 6]:
        df[f"discharge_mean_{window}h"] = df["totdcwtrqy"].rolling(window, min_periods=window).mean()

    df[INFLOW_TARGET] = df["inflowqy"].shift(-3)
    df[DISCHARGE_TARGET] = df["totdcwtrqy"].shift(-3)
    df["inflow_change_3h"] = df[INFLOW_TARGET] - df["inflowqy"]
    return df


def make_inflow_predictions(train, test):
    train = train.sort_values("obsrdt").copy()
    test = test.sort_values("obsrdt").copy()

    # 방류량 모델 학습용 예측 유입량은 시간순 OOF 방식으로 만든다.
    train["predicted_inflow_3h"] = np.nan
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, valid_idx in tscv.split(train):
        fold_train = train.iloc[train_idx]
        fold_valid = train.iloc[valid_idx]
        model = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        model.fit(fold_train[INFLOW_FEATURES], fold_train[INFLOW_TARGET])
        train.loc[fold_valid.index, "predicted_inflow_3h"] = model.predict(fold_valid[INFLOW_FEATURES])

    final_model = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    final_model.fit(train[INFLOW_FEATURES], train[INFLOW_TARGET])
    test["predicted_inflow_3h"] = final_model.predict(test[INFLOW_FEATURES])

    return train, test


def discharge_models():
    return {
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
    }


def fit_discharge_model(train, features, model):
    train_clean = train.dropna(subset=features + [DISCHARGE_TARGET]).copy()
    model.fit(train_clean[features], train_clean[DISCHARGE_TARGET])
    return model, len(train_clean)


def evaluate_fitted_model(test, feature_set_name, features, model_name, model, train_rows, eval_name, eval_mask):
    test_clean = test.loc[eval_mask].dropna(subset=features + [DISCHARGE_TARGET]).copy()

    pred = model.predict(test_clean[features])
    baseline = test_clean["totdcwtrqy"].to_numpy()

    model_metrics = metrics(test_clean[DISCHARGE_TARGET], pred)
    baseline_metrics = metrics(test_clean[DISCHARGE_TARGET], baseline)
    result = {
        "feature_set": feature_set_name,
        "model": model_name,
        "eval_scope": eval_name,
        "train_rows": train_rows,
        "test_rows": len(test_clean),
        "target_mean": test_clean[DISCHARGE_TARGET].mean(),
        "target_max": test_clean[DISCHARGE_TARGET].max(),
        "baseline_MAE": baseline_metrics["MAE"],
        "baseline_RMSE": baseline_metrics["RMSE"],
        "baseline_R2": baseline_metrics["R2"],
        **model_metrics,
    }

    pred_df = test_clean[["obsrdt", "totdcwtrqy", DISCHARGE_TARGET, "inflowqy", INFLOW_TARGET]].copy()
    pred_df["prediction"] = pred
    pred_df["baseline_prediction"] = baseline
    pred_df["predicted_inflow_3h"] = test_clean.get("predicted_inflow_3h", np.nan)
    pred_df["feature_set"] = feature_set_name
    pred_df["model"] = model_name
    pred_df["eval_scope"] = eval_name
    return result, pred_df


def main():
    df = make_data()
    needed = list(
        dict.fromkeys(
            ["obsrdt", INFLOW_TARGET, DISCHARGE_TARGET, "totdcwtrqy", "inflow_change_3h"]
            + INFLOW_FEATURES
            + BASE_DISCHARGE_FEATURES
        )
    )
    clean = df[needed].dropna().copy()

    train = clean[clean["obsrdt"] < TEST_START].copy()
    test = clean[(clean["obsrdt"] >= TEST_START) & (clean["obsrdt"] < TEST_END)].copy()

    train, test = make_inflow_predictions(train, test)

    increase_threshold = train["inflow_change_3h"].quantile(0.95)
    discharge_threshold = train[DISCHARGE_TARGET].quantile(0.95)

    eval_masks = {
        "2025_전체": pd.Series(True, index=test.index),
        "2025_강우일": test["rain_sum_24h"] > 0,
        "2025_강한강우": test["rain_sum_24h"] >= 10,
        "2025_유입량증가상위5퍼센트": test["inflow_change_3h"] >= increase_threshold,
        "2025_방류량상위5퍼센트": test[DISCHARGE_TARGET] >= discharge_threshold,
    }

    feature_sets = {
        "기본피처": BASE_DISCHARGE_FEATURES,
        "기본피처_예측유입량추가": BASE_DISCHARGE_FEATURES + ["predicted_inflow_3h"],
        # 실제 미래 유입량은 운영 시점에서는 모르는 값이므로 참고용 상한선으로만 사용한다.
        "기본피처_실제미래유입량참고용": BASE_DISCHARGE_FEATURES + [INFLOW_TARGET],
    }

    results = []
    predictions = []

    for feature_set_name, features in feature_sets.items():
        for model_name, model in discharge_models().items():
            fitted_model, train_rows = fit_discharge_model(train, features, model)
            for eval_name, eval_mask in eval_masks.items():
                if eval_mask.sum() < 20:
                    continue
                result, pred_df = evaluate_fitted_model(
                    test,
                    feature_set_name,
                    features,
                    model_name,
                    fitted_model,
                    train_rows,
                    eval_name,
                    eval_mask,
                )
                results.append(result)
                if eval_name in ["2025_유입량증가상위5퍼센트", "2025_방류량상위5퍼센트"]:
                    predictions.append(pred_df)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True)

    inflow_eval = metrics(test[INFLOW_TARGET], test["predicted_inflow_3h"])
    inflow_eval_rows = pd.DataFrame(
        [
            {
                "target": "3시간뒤유입량",
                "test_rows": len(test),
                **inflow_eval,
            }
        ]
    )

    results_df.to_csv("보령댐_예측유입량기반_방류량예측_평가결과.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv("보령댐_예측유입량기반_방류량예측_이벤트예측결과.csv", index=False, encoding="utf-8-sig")
    inflow_eval_rows.to_csv("보령댐_예측유입량기반_유입량예측성능.csv", index=False, encoding="utf-8-sig")

    print("INFLOW_PREDICTION")
    print(inflow_eval_rows.to_string(index=False))
    print("\nBEST_DISCHARGE_BY_SCOPE")
    best = results_df.sort_values("MAE").groupby(["eval_scope", "feature_set"], as_index=False).first()
    print(
        best[
            [
                "eval_scope",
                "feature_set",
                "model",
                "test_rows",
                "baseline_MAE",
                "MAE",
                "baseline_R2",
                "R2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
