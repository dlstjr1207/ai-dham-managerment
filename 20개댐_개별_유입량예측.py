from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("final_data_20_weather.csv")
HORIZON = 3
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
TARGET = "target_inflow_3h"

DAMS = {
    1012110: "소양강댐",
    1003110: "충주댐",
    1006110: "횡성댐",
    2001110: "안동댐",
    2002110: "임하댐",
    2002111: "성덕댐",
    2004101: "영주댐",
    2008101: "군위댐",
    2012101: "보현산댐",
    2015110: "합천댐",
    2018110: "남강댐",
    2021110: "밀양댐",
    3001110: "용담댐",
    3008110: "대청댐",
    4001110: "섬진강댐",
    4007110: "주암(본)댐",
    4104610: "주암(조)댐",
    3303110: "부안댐",
    3203110: "보령댐",
    5101110: "장흥댐",
}

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
    }


def add_features(df):
    df = df.sort_values("obsrdt").reset_index(drop=True)

    for lag in [1, 3, 6, 12]:
        df[f"inflow_lag_{lag}h"] = df["inflowqy"].shift(lag)
        df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f"inflow_mean_{window}h"] = (
            df["inflowqy"].rolling(window, min_periods=window).mean()
        )

    for window in [3, 6, 12, 24, 48, 72]:
        df[f"rain_sum_{window}h"] = (
            df["rain"].rolling(window, min_periods=window).sum()
        )

    df[TARGET] = df["inflowqy"].shift(-HORIZON)
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


def train_one_dam(raw, dam_code, dam_name):
    dam = raw[raw["dam_code"] == dam_code].copy()
    dam = add_features(dam)

    needed_columns = ["obsrdt", "inflowqy", TARGET] + FEATURES
    data = dam[list(dict.fromkeys(needed_columns))].dropna().copy()

    train = data[data["obsrdt"] < TRAIN_END].copy()
    test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)].copy()

    if train.empty or test.empty:
        raise ValueError(f"{dam_name}({dam_code}) 학습 또는 평가 데이터가 비어 있습니다.")

    rows = []
    prediction_frames = []

    for model_name, model in build_models().items():
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])

        rows.append(
            {
                "댐코드": dam_code,
                "댐이름": dam_name,
                "예측대상": "3시간 뒤 유입량",
                "모델": model_name,
                "학습행수": len(train),
                "평가행수": len(test),
                "평가기간_시작": test["obsrdt"].min(),
                "평가기간_끝": test["obsrdt"].max(),
                "실제평균": test[TARGET].mean(),
                "실제최대": test[TARGET].max(),
                **evaluate(test[TARGET], pred, test["inflowqy"].to_numpy()),
            }
        )

        prediction = test[["obsrdt", "inflowqy", TARGET]].copy()
        prediction["댐코드"] = dam_code
        prediction["댐이름"] = dam_name
        prediction["모델"] = model_name
        prediction["예측유입량"] = pred
        prediction["절대오차"] = (prediction[TARGET] - prediction["예측유입량"]).abs()
        prediction_frames.append(prediction)

    result = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return result, predictions


def write_report(best_results):
    lines = [
        "# 20개 댐 개별 유입량 예측 결과 보고서",
        "",
        "## 1. 실험 목적",
        "",
        "남강댐과 임하댐에서 확정한 방식대로 20개 다목적댐을 각각 분리해 3시간 뒤 유입량을 예측했다.",
        "전체 댐을 하나의 통합 모델로 묶지 않고, 댐마다 별도의 모델을 학습했다.",
        "",
        "## 2. 공통 전처리 방식",
        "",
        "각 댐별로 다음 전처리를 동일하게 적용했다.",
        "",
        "1. 해당 댐 코드만 필터링",
        "2. 관측시각 `obsrdt` 기준 시간순 정렬",
        "3. 과거 유입량 지연 변수 생성: 1시간, 3시간, 6시간, 12시간 전",
        "4. 유입량 이동평균 생성: 3시간, 6시간, 12시간, 24시간",
        "5. 강수량 지연 변수 생성: 1시간, 3시간, 6시간, 12시간 전",
        "6. 강수량 누적 변수 생성: 3시간, 6시간, 12시간, 24시간, 48시간, 72시간",
        "7. 3시간 뒤 유입량 `target_inflow_3h = inflowqy(t+3)` 생성",
        "8. 입력값 또는 목표값이 비어 있는 행 제거",
        "",
        "임의의 값 대체는 사용하지 않았다.",
        "",
        "## 3. 학습 및 평가 기간",
        "",
        "| 구분 | 기간 |",
        "|---|---|",
        "| 학습 | 2023-01-01 ~ 2024-12-31 |",
        "| 평가 | 2025-01-01 ~ 2025-12-31 |",
        "",
        "평가는 시간 순서를 지키기 위해 무작위 분할을 사용하지 않았다.",
        "",
        "## 4. 비교 모델",
        "",
        "| 모델 | 설명 |",
        "|---|---|",
        "| GradientBoosting | 순차적으로 오차를 줄이는 트리 기반 회귀 모델 |",
        "| RandomForest | 여러 결정트리를 평균내는 앙상블 회귀 모델 |",
        "",
        "기준선은 현재 유입량이 3시간 뒤에도 그대로 유지된다고 보는 단순 예측이다.",
        "",
        "```text",
        "baseline_prediction(t+3) = inflowqy(t)",
        "```",
        "",
        "## 5. 댐별 최고 모델 결과",
        "",
        "| 순위 | 댐 | 최고모델 | 모델 MAE | 기준선 MAE | 개선율(%) | 모델 R2 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]

    ranked = best_results.sort_values(["MAE_개선율(%)", "모델_R2"], ascending=[False, False])
    for idx, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(
            f"| {idx} | {row['댐이름']} | {row['모델']} | "
            f"{row['모델_MAE']:.3f} | {row['기준선_MAE']:.3f} | "
            f"{row['MAE_개선율(%)']:.2f} | {row['모델_R2']:.3f} |"
        )

    top = ranked.iloc[0]
    bottom = ranked.iloc[-1]
    lines.extend(
        [
            "",
            "## 6. 해석",
            "",
            f"가장 개선율이 큰 댐은 {top['댐이름']}이며, 최고 모델은 {top['모델']}이다.",
            f"가장 개선율이 낮은 댐은 {bottom['댐이름']}이며, 이 댐은 추가 변수나 다른 모델 구조 검토가 필요하다.",
            "",
            "이번 결과는 20개 댐을 하나로 묶은 결과가 아니라, 각 댐을 독립적으로 학습한 결과이다.",
            "따라서 이후 방류량 예측도 같은 방식으로 댐별 개별 모델을 만드는 것이 적절하다.",
        ]
    )

    Path("20개댐_개별_유입량예측_결과보고서.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    if "--report-only" in sys.argv:
        best_df = pd.read_csv("20개댐_개별_유입량예측_댐별최고모델.csv")
        write_report(best_df)
        print("저장된 평가결과로 보고서만 다시 생성했습니다.")
        return

    raw = pd.read_csv(DATA_PATH, parse_dates=["obsrdt"])

    all_results = []
    all_predictions = []

    for dam_code, dam_name in DAMS.items():
        print(f"{dam_name} 유입량 예측 학습 중...", flush=True)
        result, predictions = train_one_dam(raw, dam_code, dam_name)
        all_results.append(result)
        all_predictions.append(predictions)

    result_df = pd.concat(all_results, ignore_index=True)
    prediction_df = pd.concat(all_predictions, ignore_index=True)

    best_idx = result_df.groupby("댐코드")["모델_MAE"].idxmin()
    best_df = (
        result_df.loc[best_idx]
        .sort_values(["MAE_개선율(%)", "모델_R2"], ascending=[False, False])
        .reset_index(drop=True)
    )

    best_keys = best_df[["댐코드", "모델"]].drop_duplicates()
    best_prediction_df = prediction_df.merge(best_keys, on=["댐코드", "모델"], how="inner")

    result_df.to_csv("20개댐_개별_유입량예측_전체평가결과.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv("20개댐_개별_유입량예측_댐별최고모델.csv", index=False, encoding="utf-8-sig")
    best_prediction_df.to_csv("20개댐_개별_유입량예측_최고모델예측결과.csv", index=False, encoding="utf-8-sig")
    write_report(best_df)

    print("\n댐별 최고 모델")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
