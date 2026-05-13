from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("보령댐_운영_강수량_병합데이터.csv")
TEST_START = pd.Timestamp("2026-01-01 00:00:00")
TARGET = "inflow_mean_3h_t_plus_3h"


MODEL_FEATURES = {
    # 논문 표의 입력 노드 수 6개 구성에 맞춘 형태:
    # 현재/3시간 전 유입량 + 3/6/9/12시간 전 강수량
    "모델_I": [
        "inflow_mean_3h_t",
        "inflow_mean_3h_t_minus_3h",
        "rain_mean_3h_t_minus_3h",
        "rain_mean_3h_t_minus_6h",
        "rain_mean_3h_t_minus_9h",
        "rain_mean_3h_t_minus_12h",
    ],
    # 현재 강수량까지 포함한 7개 입력 구성
    "모델_II": [
        "inflow_mean_3h_t",
        "inflow_mean_3h_t_minus_3h",
        "rain_mean_3h_t",
        "rain_mean_3h_t_minus_3h",
        "rain_mean_3h_t_minus_6h",
        "rain_mean_3h_t_minus_9h",
        "rain_mean_3h_t_minus_12h",
    ],
    # 6시간 전 유입량까지 포함한 7개 입력 구성
    "모델_III": [
        "inflow_mean_3h_t",
        "inflow_mean_3h_t_minus_3h",
        "inflow_mean_3h_t_minus_6h",
        "rain_mean_3h_t_minus_3h",
        "rain_mean_3h_t_minus_6h",
        "rain_mean_3h_t_minus_9h",
        "rain_mean_3h_t_minus_12h",
    ],
    # 유입량 3개 + 현재/과거 강수량 5개 구성
    "모델_IV": [
        "inflow_mean_3h_t",
        "inflow_mean_3h_t_minus_3h",
        "inflow_mean_3h_t_minus_6h",
        "rain_mean_3h_t",
        "rain_mean_3h_t_minus_3h",
        "rain_mean_3h_t_minus_6h",
        "rain_mean_3h_t_minus_9h",
        "rain_mean_3h_t_minus_12h",
    ],
}


def make_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def make_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["obsrdt", "rain_base_time"])
    df = df.sort_values("obsrdt").reset_index(drop=True)

    # 논문 방식에 맞춰 1시간 자료를 3시간 평균 자료로 변환한다.
    df["inflow_mean_3h_t"] = df["inflowqy"].rolling(3, min_periods=3).mean()
    df["rain_mean_3h_t"] = df["rain_mm"].rolling(3, min_periods=3).mean()

    for lag in [3, 6, 9, 12]:
        df[f"inflow_mean_3h_t_minus_{lag}h"] = df["inflow_mean_3h_t"].shift(lag)
        df[f"rain_mean_3h_t_minus_{lag}h"] = df["rain_mean_3h_t"].shift(lag)

    df[TARGET] = df["inflow_mean_3h_t"].shift(-3)
    return df


def train_model(model_name, features, hidden_layer_size, train, test):
    model = Pipeline(
        steps=[
            ("scale_x", MinMaxScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(hidden_layer_size,),
                    activation="logistic",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=2000,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=30,
                ),
            ),
        ]
    )

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(train[[TARGET]]).ravel()

    model.fit(train[features], y_train_scaled)
    pred_scaled = model.predict(test[features])
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    metrics = make_metrics(test[TARGET], pred)
    return model, pred, metrics


def main():
    df = make_dataset()

    results = []
    all_predictions = []

    for model_name, features in MODEL_FEATURES.items():
        needed = ["obsrdt", TARGET, "inflow_mean_3h_t"] + features
        needed = list(dict.fromkeys(needed))
        clean = df[needed].dropna().copy()

        train = clean[clean["obsrdt"] < TEST_START]
        test = clean[clean["obsrdt"] >= TEST_START]

        input_count = len(features)
        hidden_candidates = sorted(set([input_count, int(input_count * 1.5), input_count * 2]))

        best = None
        for hidden in hidden_candidates:
            _, pred, metrics = train_model(model_name, features, hidden, train, test)

            current_inflow_pred = test["inflow_mean_3h_t"].to_numpy()
            baseline = make_metrics(test[TARGET], current_inflow_pred)

            result = {
                "model": model_name,
                "hidden_nodes": hidden,
                "input_count": input_count,
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
                "current_inflow_baseline_MAE": baseline["MAE"],
                "current_inflow_baseline_RMSE": baseline["RMSE"],
                "current_inflow_baseline_R2": baseline["R2"],
                **metrics,
            }
            results.append(result)

            if best is None or metrics["MAE"] < best["metrics"]["MAE"]:
                best = {
                    "hidden_nodes": hidden,
                    "pred": pred,
                    "metrics": metrics,
                    "baseline": baseline,
                }

        pred_df = test[["obsrdt", TARGET, "inflow_mean_3h_t"]].copy()
        pred_df["prediction"] = best["pred"]
        pred_df["error"] = pred_df["prediction"] - pred_df[TARGET]
        pred_df["model"] = model_name
        pred_df["hidden_nodes"] = best["hidden_nodes"]
        all_predictions.append(pred_df)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_df.to_csv("보령댐_유입량_신경망_평가결과.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv("보령댐_유입량_신경망_예측결과.csv", index=False, encoding="utf-8-sig")

    print("RESULTS")
    print(results_df.sort_values(["model", "MAE"]).to_string(index=False))

    print("\nBEST_BY_MODEL")
    best_rows = results_df.sort_values("MAE").groupby("model", as_index=False).first()
    print(best_rows[["model", "hidden_nodes", "MAE", "RMSE", "R2", "current_inflow_baseline_MAE", "current_inflow_baseline_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
