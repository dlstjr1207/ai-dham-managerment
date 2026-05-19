from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("final_data_20_weather.csv")
GRADE_PATH = Path("20개 댐 유입량 예측") / "20개댐_유입량예측_성능등급표.csv"
OUTPUT_DIR = Path("후보댐 방류량 예측")
OUTPUT_DIR.mkdir(exist_ok=True)

HORIZON = 3
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00")
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
            n_estimators=50,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
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
        df[f"inflow_mean_{window}h"] = df["inflowqy"].rolling(window, min_periods=window).mean()
        df[f"discharge_mean_{window}h"] = df["totdcwtrqy"].rolling(window, min_periods=window).mean()

    for window in [3, 6, 12, 24, 48, 72]:
        df[f"rain_sum_{window}h"] = df["rain"].rolling(window, min_periods=window).sum()

    df["discharge_change_lag_1h"] = df["totdcwtrqy"].diff(1)
    df["discharge_change_lag_3h"] = df["totdcwtrqy"] - df["totdcwtrqy"].shift(3)
    df[INFLOW_TARGET] = df["inflowqy"].shift(-HORIZON)
    df[DISCHARGE_TARGET] = df["totdcwtrqy"].shift(-HORIZON)
    df["inflow_change_3h"] = df[INFLOW_TARGET] - df["inflowqy"]
    df["discharge_change_3h"] = df[DISCHARGE_TARGET] - df["totdcwtrqy"]
    return df


def metric_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def improvement(baseline_mae, model_mae):
    if baseline_mae == 0:
        return 0
    return (baseline_mae - model_mae) / baseline_mae * 100


def train_best_inflow(train, test):
    rows = []
    fitted = {}
    for model_name, model in build_models().items():
        model.fit(train[INFLOW_FEATURES], train[INFLOW_TARGET])
        pred = model.predict(test[INFLOW_FEATURES])
        rows.append({"유입량예측모델": model_name, **metric_dict(test[INFLOW_TARGET], pred)})
        fitted[model_name] = model

    ranking = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    best_name = ranking.iloc[0]["유입량예측모델"]
    return best_name, fitted[best_name], ranking


def evaluate_overall(test, pred):
    baseline_mae = mean_absolute_error(test[DISCHARGE_TARGET], test["totdcwtrqy"])
    model_metrics = metric_dict(test[DISCHARGE_TARGET], pred)
    return {
        "전체_기준선_MAE": baseline_mae,
        "전체_모델_MAE": model_metrics["MAE"],
        "전체_개선율(%)": improvement(baseline_mae, model_metrics["MAE"]),
        "전체_모델_R2": model_metrics["R2"],
    }


def evaluate_final_standard(test, pred, dam_code, dam_name, feature_set, discharge_model, inflow_model):
    base = test.copy()
    base["예측방류량"] = pred
    inflow_threshold = base["inflow_change_3h"].quantile(0.9)

    masks = {
        "전체": pd.Series(True, index=base.index),
        "강우발생": base["rain"] > 0,
        "최근24시간누적강우": base["rain_sum_24h"] > 0,
        "유입량급증": base["inflow_change_3h"] >= inflow_threshold,
        "방류량변화": base["discharge_change_3h"].abs() >= 5,
    }
    baselines = {
        "현재방류량유지": "totdcwtrqy",
        "최근3시간평균": "discharge_mean_3h",
        "최근6시간평균": "discharge_mean_6h",
        "최근3시간변화량반영": "discharge_trend_3h",
    }
    base["discharge_trend_3h"] = (base["totdcwtrqy"] + base["discharge_change_lag_3h"]).clip(lower=0)

    rows = []
    for segment_name, mask in masks.items():
        subset = base[mask].dropna(subset=[DISCHARGE_TARGET, "예측방류량"] + list(baselines.values()))
        if subset.empty:
            continue
        model_mae = mean_absolute_error(subset[DISCHARGE_TARGET], subset["예측방류량"])
        for baseline_name, column in baselines.items():
            baseline_mae = mean_absolute_error(subset[DISCHARGE_TARGET], subset[column])
            rows.append(
                {
                    "댐코드": dam_code,
                    "댐이름": dam_name,
                    "입력변수세트": feature_set,
                    "방류량모델": discharge_model,
                    "유입량예측모델": inflow_model,
                    "평가구간": segment_name,
                    "평가건수": len(subset),
                    "기준선": baseline_name,
                    "기준선_MAE": baseline_mae,
                    "모델_MAE": model_mae,
                    "개선율(%)": improvement(baseline_mae, model_mae),
                }
            )
    return rows


def train_one(raw, dam_code, dam_name, inflow_grade):
    dam = add_features(raw[raw["dam_code"] == dam_code].copy())
    needed = list(
        dict.fromkeys(
            ["obsrdt", "inflowqy", "totdcwtrqy", INFLOW_TARGET, DISCHARGE_TARGET, "inflow_change_3h", "discharge_change_3h"]
            + INFLOW_FEATURES
            + DISCHARGE_BASE_FEATURES
        )
    )
    data = dam[needed].dropna().copy()
    train = data[data["obsrdt"] < TRAIN_END].copy()
    test = data[(data["obsrdt"] >= TRAIN_END) & (data["obsrdt"] < TEST_END)].copy()

    best_inflow_name, best_inflow_model, inflow_ranking = train_best_inflow(train, test)
    train["predicted_inflow_3h"] = best_inflow_model.predict(train[INFLOW_FEATURES])
    test["predicted_inflow_3h"] = best_inflow_model.predict(test[INFLOW_FEATURES])

    feature_sets = {
        "기본변수": DISCHARGE_BASE_FEATURES,
        "예측유입량포함": DISCHARGE_BASE_FEATURES + ["predicted_inflow_3h"],
    }

    overall_rows = []
    final_rows = []
    prediction_frames = []
    for feature_set, features in feature_sets.items():
        for model_name, model in build_models().items():
            model.fit(train[features], train[DISCHARGE_TARGET])
            pred = model.predict(test[features])
            overall_rows.append(
                {
                    "댐코드": dam_code,
                    "댐이름": dam_name,
                    "유입량성능등급": inflow_grade,
                    "입력변수세트": feature_set,
                    "방류량모델": model_name,
                    "유입량예측모델": best_inflow_name,
                    "학습행수": len(train),
                    "평가행수": len(test),
                    **evaluate_overall(test, pred),
                }
            )
            final_rows.extend(
                evaluate_final_standard(
                    test, pred, dam_code, dam_name, feature_set, model_name, best_inflow_name
                )
            )

            pred_df = test[["obsrdt", "totdcwtrqy", DISCHARGE_TARGET]].copy()
            pred_df["댐코드"] = dam_code
            pred_df["댐이름"] = dam_name
            pred_df["유입량성능등급"] = inflow_grade
            pred_df["입력변수세트"] = feature_set
            pred_df["방류량모델"] = model_name
            pred_df["유입량예측모델"] = best_inflow_name
            pred_df["예측방류량"] = pred
            prediction_frames.append(pred_df)

    inflow_ranking["댐코드"] = dam_code
    inflow_ranking["댐이름"] = dam_name
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(final_rows),
        pd.concat(prediction_frames, ignore_index=True),
        inflow_ranking,
    )


def write_report(best, final_eval):
    lines = [
        "# 후보댐 방류량 예측 확장 보고서",
        "",
        "## 1. 목적",
        "",
        "유입량 예측에서 우수 또는 양호 등급으로 분류된 댐을 대상으로 방류량 예측을 확장했다.",
        "평가는 전체 기간 MAE 하나로 끝내지 않고, 새로 정리한 이벤트 중심 평가 기준을 함께 적용했다.",
        "",
        "## 2. 대상",
        "",
        "대상은 유입량 예측 성능등급이 `우수` 또는 `양호`인 댐이다.",
        "",
        "## 3. 전체 기간 기준 최고 모델",
        "",
        "| 댐 | 유입량등급 | 입력변수세트 | 방류량모델 | 유입량예측모델 | 기준선 MAE | 모델 MAE | 개선율(%) | R2 |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['유입량성능등급']} | {row['입력변수세트']} | {row['방류량모델']} | "
            f"{row['유입량예측모델']} | {row['전체_기준선_MAE']:.3f} | {row['전체_모델_MAE']:.3f} | "
            f"{row['전체_개선율(%)']:.2f} | {row['전체_모델_R2']:.3f} |"
        )

    focus = final_eval[
        (final_eval["기준선"] == "현재방류량유지")
        & (final_eval["평가구간"] == "방류량변화")
    ].copy()
    best_keys = best[["댐코드", "입력변수세트", "방류량모델"]]
    focus = focus.merge(best_keys, on=["댐코드", "입력변수세트", "방류량모델"], how="inner")
    focus = focus.sort_values("개선율(%)", ascending=False)

    lines.extend(
        [
            "",
            "## 4. 방류량 변화 구간 평가",
            "",
            "방류량 예측의 핵심은 평상시 유지 구간보다 실제 방류량이 변하는 구간을 얼마나 잘 잡는지이다.",
            "",
            "| 댐 | 평가건수 | 기준선 MAE | 모델 MAE | 개선율(%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in focus.iterrows():
        lines.append(
            f"| {row['댐이름']} | {int(row['평가건수'])} | {row['기준선_MAE']:.3f} | "
            f"{row['모델_MAE']:.3f} | {row['개선율(%)']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 해석",
            "",
            "전체 기간에서 기준선을 이기는 댐은 방류량 예측을 바로 적용할 가능성이 크다.",
            "전체 기간에서는 부족하더라도 방류량 변화 구간에서 개선되는 댐은 이벤트 중심 보조 모델로 의미가 있다.",
            "둘 다 개선되지 않는 댐은 운영 규칙, 계절, 저수율 구간 등 추가 변수가 필요하다.",
        ]
    )
    (OUTPUT_DIR / "후보댐_방류량예측_확장보고서.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    grade = pd.read_csv(GRADE_PATH)
    candidates = grade[grade["성능등급"].isin(["우수", "양호"])][["댐코드", "댐이름", "성능등급"]]
    raw = pd.read_csv(DATA_PATH, parse_dates=["obsrdt"])

    all_overall = []
    all_final = []
    all_predictions = []
    all_inflow_rankings = []
    for row in candidates.itertuples(index=False):
        print(f"{row.댐이름} 방류량 예측 확장 학습 중...", flush=True)
        overall, final_eval, predictions, inflow_ranking = train_one(raw, row.댐코드, row.댐이름, row.성능등급)
        all_overall.append(overall)
        all_final.append(final_eval)
        all_predictions.append(predictions)
        all_inflow_rankings.append(inflow_ranking)

    overall_df = pd.concat(all_overall, ignore_index=True)
    final_df = pd.concat(all_final, ignore_index=True)
    prediction_df = pd.concat(all_predictions, ignore_index=True)
    inflow_ranking_df = pd.concat(all_inflow_rankings, ignore_index=True)

    best_idx = overall_df.groupby("댐코드")["전체_모델_MAE"].idxmin()
    best_df = (
        overall_df.loc[best_idx]
        .sort_values("전체_개선율(%)", ascending=False)
        .reset_index(drop=True)
    )

    best_keys = best_df[["댐코드", "입력변수세트", "방류량모델"]]
    best_predictions = prediction_df.merge(best_keys, on=["댐코드", "입력변수세트", "방류량모델"], how="inner")

    overall_df.to_csv(OUTPUT_DIR / "후보댐_방류량예측_전체평가결과.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUTPUT_DIR / "후보댐_방류량예측_댐별최고모델.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(OUTPUT_DIR / "후보댐_방류량예측_새평가기준결과.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(OUTPUT_DIR / "후보댐_방류량예측_최고모델예측결과.csv", index=False, encoding="utf-8-sig")
    inflow_ranking_df.to_csv(OUTPUT_DIR / "후보댐_방류량예측용_유입량모델평가.csv", index=False, encoding="utf-8-sig")
    write_report(best_df, final_df)

    print("\n전체 기간 기준 댐별 최고 모델")
    print(best_df[["댐이름", "유입량성능등급", "입력변수세트", "방류량모델", "전체_기준선_MAE", "전체_모델_MAE", "전체_개선율(%)", "전체_모델_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
