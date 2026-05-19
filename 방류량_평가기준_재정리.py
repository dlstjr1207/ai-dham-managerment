from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error


RAW_DATA_PATH = Path("final_data_20_weather.csv")
PREDICTION_PATH = Path("우수댐 방류량 예측") / "우수댐_방류량예측_최고모델예측결과.csv"
OUTPUT_DIR = Path("우수댐 방류량 예측")
OUTPUT_CSV = OUTPUT_DIR / "방류량_평가기준_재정리_결과.csv"
REPORT_PATH = OUTPUT_DIR / "방류량_평가기준_재정리_보고서.md"

HORIZON = 3


def add_context_features(df):
    df = df.sort_values(["dam_code", "obsrdt"]).copy()
    grouped = df.groupby("dam_code", group_keys=False)

    df["target_inflow_3h"] = grouped["inflowqy"].shift(-HORIZON)
    df["target_discharge_3h_check"] = grouped["totdcwtrqy"].shift(-HORIZON)
    df["discharge_mean_3h"] = grouped["totdcwtrqy"].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    df["discharge_mean_6h"] = grouped["totdcwtrqy"].rolling(6, min_periods=6).mean().reset_index(level=0, drop=True)
    df["discharge_trend_3h"] = df["totdcwtrqy"] + (df["totdcwtrqy"] - grouped["totdcwtrqy"].shift(3))
    df["rain_sum_6h"] = grouped["rain"].rolling(6, min_periods=6).sum().reset_index(level=0, drop=True)
    df["rain_sum_24h"] = grouped["rain"].rolling(24, min_periods=24).sum().reset_index(level=0, drop=True)
    df["inflow_change_3h"] = df["target_inflow_3h"] - df["inflowqy"]
    df["discharge_change_3h"] = df["target_discharge_3h_check"] - df["totdcwtrqy"]
    return df


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def improvement(baseline_mae, model_mae):
    if baseline_mae == 0:
        return 0
    return (baseline_mae - model_mae) / baseline_mae * 100


def evaluate_subset(df, subset_name, mask, inflow_threshold):
    subset = df[mask].dropna(
        subset=[
            "target_discharge_3h",
            "예측방류량",
            "totdcwtrqy",
            "discharge_mean_3h",
            "discharge_mean_6h",
            "discharge_trend_3h",
        ]
    )
    if subset.empty:
        return []

    baselines = {
        "현재방류량유지": subset["totdcwtrqy"],
        "최근3시간평균": subset["discharge_mean_3h"],
        "최근6시간평균": subset["discharge_mean_6h"],
        "최근3시간변화량반영": subset["discharge_trend_3h"].clip(lower=0),
    }

    rows = []
    model_mae = mae(subset["target_discharge_3h"], subset["예측방류량"])
    for baseline_name, baseline_pred in baselines.items():
        baseline_mae = mae(subset["target_discharge_3h"], baseline_pred)
        rows.append(
            {
                "댐코드": subset["댐코드"].iloc[0],
                "댐이름": subset["댐이름"].iloc[0],
                "평가구간": subset_name,
                "평가건수": len(subset),
                "유입급증기준값": inflow_threshold,
                "기준선": baseline_name,
                "기준선_MAE": baseline_mae,
                "모델_MAE": model_mae,
                "개선율(%)": improvement(baseline_mae, model_mae),
            }
        )
    return rows


def build_report(result):
    lines = [
        "# 방류량 예측 평가 기준 재정리 보고서",
        "",
        "## 1. 재정리 이유",
        "",
        "방류량은 많은 시간 동안 현재 값이 유지되는 특성이 있어 `현재 방류량 유지` 기준선이 매우 강하게 작동한다.",
        "따라서 전체 기간 하나의 MAE만으로 모델을 판단하면, 실제로 중요한 강우 이후 구간이나 방류량 변화 구간의 예측 성능이 가려질 수 있다.",
        "",
        "이번 보고서에서는 기준선을 낮추지 않고, 평가 구간과 기준선을 더 현실적으로 나누어 비교했다.",
        "",
        "## 2. 새 평가 기준",
        "",
        "### 평가 구간",
        "",
        "| 구간 | 의미 |",
        "|---|---|",
        "| 전체 | 2025년 평가기간 전체 |",
        "| 강우발생 | 현재 시점 강수량이 0보다 큰 구간 |",
        "| 최근24시간누적강우 | 최근 24시간 누적 강수량이 0보다 큰 구간 |",
        "| 유입량급증 | 3시간 뒤 유입량 증가량이 댐별 상위 10% 이상인 구간 |",
        "| 방류량변화 | 3시간 뒤 방류량이 현재보다 5 이상 변한 구간 |",
        "",
        "### 비교 기준선",
        "",
        "| 기준선 | 계산식 |",
        "|---|---|",
        "| 현재방류량유지 | `totdcwtrqy(t+3) = totdcwtrqy(t)` |",
        "| 최근3시간평균 | 최근 3시간 방류량 평균 |",
        "| 최근6시간평균 | 최근 6시간 방류량 평균 |",
        "| 최근3시간변화량반영 | 현재 방류량 + 최근 3시간 변화량 |",
        "",
        "## 3. 핵심 결과",
        "",
        "| 댐 | 평가구간 | 기준선 | 기준선 MAE | 모델 MAE | 개선율(%) |",
        "|---|---|---|---:|---:|---:|",
    ]

    focus = result[
        (result["기준선"] == "현재방류량유지")
        & (result["평가구간"].isin(["전체", "최근24시간누적강우", "유입량급증", "방류량변화"]))
    ].copy()
    for _, row in focus.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['평가구간']} | {row['기준선']} | "
            f"{row['기준선_MAE']:.3f} | {row['모델_MAE']:.3f} | {row['개선율(%)']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 해석 방법",
            "",
            "방류량 예측에서는 전체 기간 MAE보다 `방류량변화`, `유입량급증`, `최근24시간누적강우` 구간의 결과를 함께 봐야 한다.",
            "전체 기간에서 기준선보다 낮지 않더라도 방류량 변화 구간에서 개선된다면, 모델은 운영 변화 시점 탐지에 의미가 있을 수 있다.",
            "",
            "반대로 강우나 유입량 급증 구간에서도 기준선을 이기지 못하면 현재 입력 변수만으로는 방류량 운영 판단을 설명하기 어렵다고 봐야 한다.",
            "",
            "## 5. 앞으로 적용할 평가 원칙",
            "",
            "1. 방류량 예측은 전체 기간 MAE 하나로 결론내리지 않는다.",
            "2. 현재방류량유지 기준선은 유지하되, 최근 평균과 변화량 기준선도 함께 둔다.",
            "3. 강우, 누적강우, 유입량 급증, 방류량 변화 구간을 별도로 평가한다.",
            "4. 최종 성공 판단은 전체 성능과 이벤트 구간 성능을 함께 보고 결정한다.",
        ]
    )
    return "\n".join(lines)


def main():
    raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["obsrdt"])
    predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["obsrdt"])

    context = add_context_features(raw)
    context_cols = [
        "dam_code",
        "obsrdt",
        "rain",
        "rain_sum_24h",
        "inflow_change_3h",
        "discharge_change_3h",
        "discharge_mean_3h",
        "discharge_mean_6h",
        "discharge_trend_3h",
    ]
    merged = predictions.merge(
        context[context_cols],
        left_on=["댐코드", "obsrdt"],
        right_on=["dam_code", "obsrdt"],
        how="left",
    )

    rows = []
    for _, dam_df in merged.groupby("댐코드"):
        inflow_threshold = dam_df["inflow_change_3h"].quantile(0.9)
        masks = {
            "전체": pd.Series(True, index=dam_df.index),
            "강우발생": dam_df["rain"] > 0,
            "최근24시간누적강우": dam_df["rain_sum_24h"] > 0,
            "유입량급증": dam_df["inflow_change_3h"] >= inflow_threshold,
            "방류량변화": dam_df["discharge_change_3h"].abs() >= 5,
        }
        for subset_name, mask in masks.items():
            rows.extend(evaluate_subset(dam_df, subset_name, mask, inflow_threshold))

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(result), encoding="utf-8")

    summary = result[
        (result["기준선"] == "현재방류량유지")
        & (result["평가구간"].isin(["전체", "최근24시간누적강우", "유입량급증", "방류량변화"]))
    ]
    print(summary[["댐이름", "평가구간", "평가건수", "기준선_MAE", "모델_MAE", "개선율(%)"]].to_string(index=False))


if __name__ == "__main__":
    main()
