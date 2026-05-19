from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("final_data_20_weather.csv")
OUTPUT_DIR = Path("우수댐 방류량 예측")
OUTPUT_DIR.mkdir(exist_ok=True)

HORIZON = 3
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
INFLOW_TARGET = "target_inflow_3h"
DISCHARGE_TARGET = "target_discharge_3h"

EXCELLENT_DAMS = {
    2018110: "남강댐",
    4001110: "섬진강댐",
    3001110: "용담댐",
    2015110: "합천댐",
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


def build_models():
    return {
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
    }


def add_features(df):
    df = df.sort_values("obsrdt").reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
        df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
        df[f"discharge_lag_{lag}h"] = df["totdcwtrqy"].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = (
            df["inflowqy"].rolling(window, min_periods=window).mean()
        )
        df[f"discharge_mean_{window}h"] = (
            df["totdcwtrqy"].rolling(window, min_periods=window).mean()
        )

    for window in [3, 6, 12, 24, 48, 72]:
        df[f"rain_sum_{window}h"] = (
            df["rain"].rolling(window, min_periods=window).sum()
        )

    df["discharge_change_lag_1h"] = df["totdcwtrqy"].diff(1)
    df["discharge_change_lag_3h"] = df["totdcwtrqy"] - df["totdcwtrqy"].shift(3)
    df[INFLOW_TARGET] = df["inflowqy"].shift(-HORIZON)
    df[DISCHARGE_TARGET] = df["totdcwtrqy"].shift(-HORIZON)
    return df


def metric_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def evaluate(y_true, y_pred, baseline_pred):
    baseline_metrics = metric_dict(y_true, baseline_pred)
    model_metrics = metric_dict(y_true, y_pred)
    improvement_rate = (
        (baseline_metrics["MAE"] - model_metrics["MAE"])
        / baseline_metrics["MAE"]
        * 100
        if baseline_metrics["MAE"] != 0
        else 0
    )
    return {
        "기준선_MAE": baseline_metrics["MAE"],
        "기준선_RMSE": baseline_metrics["RMSE"],
        "기준선_R2": baseline_metrics["R2"],
        "모델_MAE": model_metrics["MAE"],
        "모델_RMSE": model_metrics["RMSE"],
        "모델_R2": model_metrics["R2"],
        "MAE_개선율(%)": improvement_rate,
    }


def train_best_inflow_model(train, test):
    rows = []
    fitted_models = {}
    for model_name, model in build_models().items():
        model.fit(train[INFLOW_FEATURES], train[INFLOW_TARGET])
        pred = model.predict(test[INFLOW_FEATURES])
        rows.append(
            {
                "모델": model_name,
                **metric_dict(test[INFLOW_TARGET], pred),
            }
        )
        fitted_models[model_name] = model

    ranking = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    best_model_name = ranking.iloc[0]["모델"]
    best_model = fitted_models[best_model_name]
    return best_model_name, best_model, ranking


def event_metrics(test, pred, threshold):
    event_mask = test[DISCHARGE_TARGET] >= threshold
    change_mask = (test[DISCHARGE_TARGET] - test["totdcwtrqy"]).abs() >= 5

    def safe_mae(mask):
        if mask.sum() == 0:
            return np.nan
        return mean_absolute_error(test.loc[mask, DISCHARGE_TARGET], pred[mask])

    return {
        "고방류_기준값": threshold,
        "고방류_건수": int(event_mask.sum()),
        "고방류_MAE": safe_mae(event_mask),
        "방류변화_건수": int(change_mask.sum()),
        "방류변화_MAE": safe_mae(change_mask),
    }


def train_one_dam(raw, dam_code, dam_name):
    dam = add_features(raw[raw["dam_code"] == dam_code].copy())
    needed = list(
        dict.fromkeys(
            ["obsrdt", "inflowqy", "totdcwtrqy", INFLOW_TARGET, DISCHARGE_TARGET]
            + INFLOW_FEATURES
            + DISCHARGE_BASE_FEATURES
        )
    )
    data = dam[needed].dropna().copy()
    train = data[data["obsrdt"] < TRAIN_END].copy()
    test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)].copy()

    best_inflow_name, best_inflow_model, inflow_ranking = train_best_inflow_model(train, test)
    train["predicted_inflow_3h"] = best_inflow_model.predict(train[INFLOW_FEATURES])
    test["predicted_inflow_3h"] = best_inflow_model.predict(test[INFLOW_FEATURES])

    feature_sets = {
        "기본변수": DISCHARGE_BASE_FEATURES,
        "예측유입량포함": DISCHARGE_BASE_FEATURES + ["predicted_inflow_3h"],
        "실제미래유입량_참고용": DISCHARGE_BASE_FEATURES + [INFLOW_TARGET],
    }

    rows = []
    event_rows = []
    prediction_frames = []
    high_discharge_threshold = train[DISCHARGE_TARGET].quantile(0.9)

    for feature_set_name, features in feature_sets.items():
        for model_name, model in build_models().items():
            model.fit(train[features], train[DISCHARGE_TARGET])
            pred = model.predict(test[features])
            deployable = feature_set_name != "실제미래유입량_참고용"

            row = {
                "댐코드": dam_code,
                "댐이름": dam_name,
                "방류량_모델": model_name,
                "입력변수세트": feature_set_name,
                "실사용가능여부": "가능" if deployable else "불가_참고용",
                "유입량예측모델": best_inflow_name,
                "학습행수": len(train),
                "평가행수": len(test),
                "실제평균": test[DISCHARGE_TARGET].mean(),
                "실제최대": test[DISCHARGE_TARGET].max(),
                **evaluate(test[DISCHARGE_TARGET], pred, test["totdcwtrqy"].to_numpy()),
            }
            rows.append(row)
            event_rows.append({**row, **event_metrics(test, pred, high_discharge_threshold)})

            pred_out = test[["obsrdt", "totdcwtrqy", DISCHARGE_TARGET]].copy()
            pred_out["댐코드"] = dam_code
            pred_out["댐이름"] = dam_name
            pred_out["방류량_모델"] = model_name
            pred_out["입력변수세트"] = feature_set_name
            pred_out["유입량예측모델"] = best_inflow_name
            pred_out["예측방류량"] = pred
            pred_out["절대오차"] = (pred_out[DISCHARGE_TARGET] - pred_out["예측방류량"]).abs()
            pred_out["예측유입량_3시간뒤"] = test["predicted_inflow_3h"].to_numpy()
            prediction_frames.append(pred_out)

    inflow_ranking["댐코드"] = dam_code
    inflow_ranking["댐이름"] = dam_name
    return (
        pd.DataFrame(rows),
        pd.DataFrame(event_rows),
        pd.concat(prediction_frames, ignore_index=True),
        inflow_ranking,
    )


def write_report(best, all_results, event_best, inflow_rankings):
    lines = [
        "# 우수 등급 댐 방류량 예측 분석 보고서",
        "",
        "## 1. 분석 목적",
        "",
        "20개 댐 유입량 예측에서 우수 등급으로 분류된 남강댐, 섬진강댐, 용담댐, 합천댐을 대상으로 3시간 뒤 방류량 예측을 수행했다.",
        "방류량은 자연현상뿐 아니라 운영 판단이 반영되는 값이므로, 단순히 강수량만으로 설명하지 않고 현재 방류량, 과거 방류량, 유입량, 예측 유입량, 저수상태를 함께 사용했다.",
        "",
        "## 2. 실험 구조",
        "",
        "각 댐은 전체 데이터를 합치지 않고 개별로 학습했다.",
        "",
        "```text",
        "댐별 데이터 필터링",
        "→ 유입량 예측 모델 학습",
        "→ 예측 유입량 생성",
        "→ 방류량 예측 모델 학습",
        "→ 기준선과 성능 비교",
        "```",
        "",
        "방류량 예측 목표는 다음과 같다.",
        "",
        "```text",
        "target_discharge_3h(t) = totdcwtrqy(t + 3)",
        "```",
        "",
        "## 3. 입력 변수 세트",
        "",
        "| 입력변수세트 | 의미 | 실사용 가능 여부 |",
        "|---|---|---|",
        "| 기본변수 | 현재/과거 방류량, 현재/과거 유입량, 강수량, 저수상태 | 가능 |",
        "| 예측유입량포함 | 기본변수 + 유입량 모델이 예측한 3시간 뒤 유입량 | 가능 |",
        "| 실제미래유입량_참고용 | 기본변수 + 실제 3시간 뒤 유입량 | 불가, 참고용 상한선 |",
        "",
        "실제 미래 유입량은 예측 시점에 알 수 없는 값이므로 최종 모델 후보로 사용하지 않았다.",
        "다만 유입량을 완벽하게 알았을 때 방류량 예측이 얼마나 좋아질 수 있는지 확인하기 위한 참고용으로만 비교했다.",
        "",
        "## 4. 댐별 최고 실사용 모델",
        "",
        "| 댐 | 입력변수세트 | 방류량 모델 | 유입량예측모델 | 기준선 MAE | 모델 MAE | 개선율(%) | R2 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    for _, row in best.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량_모델']} | "
            f"{row['유입량예측모델']} | {row['기준선_MAE']:.3f} | "
            f"{row['모델_MAE']:.3f} | {row['MAE_개선율(%)']:.2f} | {row['모델_R2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 고방류 및 변화 구간 평가",
            "",
            "방류량 평균 오차만 보면 평상시 구간이 성능을 과하게 좋게 보이게 만들 수 있으므로, 고방류 구간과 방류 변화 구간의 MAE도 함께 확인했다.",
            "",
            "| 댐 | 모델 | 고방류 기준값 | 고방류 건수 | 고방류 MAE | 방류변화 건수 | 방류변화 MAE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in event_best.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['방류량_모델']} | {row['고방류_기준값']:.3f} | "
            f"{int(row['고방류_건수'])} | {row['고방류_MAE']:.3f} | "
            f"{int(row['방류변화_건수'])} | {row['방류변화_MAE']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 6. 해석",
            "",
        ]
    )

    for _, row in best.iterrows():
        if row["MAE_개선율(%)"] > 20:
            comment = "기준선보다 방류량 예측 오차가 확실히 줄어 방류량 예측 연결 가능성이 높다."
        elif row["MAE_개선율(%)"] > 0:
            comment = "기준선보다 나아지긴 했지만 운영 판단이 섞이는 방류량 특성상 추가 검증이 필요하다."
        else:
            comment = "현재 방식으로는 기준선보다 나아졌다고 보기 어려워 방류량 예측 입력 구조를 다시 점검해야 한다."
        lines.append(f"- {row['댐이름']}: {comment}")

    lines.extend(
        [
            "",
            "## 7. 다음 단계",
            "",
            "1. 성능이 좋은 댐은 예측 결과 그래프를 만들어 실제 방류량과 예측 방류량의 시간 흐름을 비교한다.",
            "2. 고방류 구간에서 오차가 큰 댐은 이벤트 중심으로 별도 분석한다.",
            "3. 실사용 가능한 입력변수세트와 참고용 입력변수세트의 차이가 큰 댐은 유입량 예측 고도화가 방류량 성능 개선에 중요하다고 해석한다.",
        ]
    )

    report_path = OUTPUT_DIR / "우수댐_방류량예측_분석보고서.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["obsrdt"])

    results = []
    events = []
    predictions = []
    inflow_rankings = []

    for dam_code, dam_name in EXCELLENT_DAMS.items():
        print(f"{dam_name} 방류량 예측 학습 중...", flush=True)
        result, event_result, prediction, inflow_ranking = train_one_dam(raw, dam_code, dam_name)
        results.append(result)
        events.append(event_result)
        predictions.append(prediction)
        inflow_rankings.append(inflow_ranking)

    result_df = pd.concat(results, ignore_index=True)
    event_df = pd.concat(events, ignore_index=True)
    prediction_df = pd.concat(predictions, ignore_index=True)
    inflow_ranking_df = pd.concat(inflow_rankings, ignore_index=True)

    deployable = result_df[result_df["실사용가능여부"] == "가능"].copy()
    best_idx = deployable.groupby("댐코드")["모델_MAE"].idxmin()
    best_df = (
        deployable.loc[best_idx]
        .sort_values("MAE_개선율(%)", ascending=False)
        .reset_index(drop=True)
    )

    best_keys = best_df[["댐코드", "방류량_모델", "입력변수세트"]]
    best_prediction_df = prediction_df.merge(
        best_keys,
        on=["댐코드", "방류량_모델", "입력변수세트"],
        how="inner",
    )
    event_best = event_df.merge(
        best_keys,
        on=["댐코드", "방류량_모델", "입력변수세트"],
        how="inner",
    )

    result_df.to_csv(OUTPUT_DIR / "우수댐_방류량예측_전체평가결과.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUTPUT_DIR / "우수댐_방류량예측_댐별최고모델.csv", index=False, encoding="utf-8-sig")
    event_df.to_csv(OUTPUT_DIR / "우수댐_방류량예측_이벤트평가결과.csv", index=False, encoding="utf-8-sig")
    event_best.to_csv(OUTPUT_DIR / "우수댐_방류량예측_최고모델_이벤트평가.csv", index=False, encoding="utf-8-sig")
    best_prediction_df.to_csv(OUTPUT_DIR / "우수댐_방류량예측_최고모델예측결과.csv", index=False, encoding="utf-8-sig")
    inflow_ranking_df.to_csv(OUTPUT_DIR / "우수댐_방류량예측용_유입량모델평가.csv", index=False, encoding="utf-8-sig")

    write_report(best_df, result_df, event_best, inflow_ranking_df)

    print("\n댐별 최고 실사용 방류량 모델")
    print(
        best_df[
            [
                "댐이름",
                "입력변수세트",
                "방류량_모델",
                "유입량예측모델",
                "기준선_MAE",
                "모델_MAE",
                "MAE_개선율(%)",
                "모델_R2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
