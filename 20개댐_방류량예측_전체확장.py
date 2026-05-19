from pathlib import Path
import importlib.util

import pandas as pd


BASE_SCRIPT = Path("후보댐_방류량예측_확장.py")
OUTPUT_DIR = Path("20개댐 방류량 예측")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_base_module():
    spec = importlib.util.spec_from_file_location("candidate_discharge", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(best_df, final_df):
    event = final_df[
        (final_df["평가구간"] == "방류량변화")
        & (final_df["기준선"] == "현재방류량유지")
    ].copy()
    event_idx = event.groupby("댐코드")["개선율(%)"].idxmax()
    event_best = (
        event.loc[event_idx]
        .sort_values("개선율(%)", ascending=False)
        .reset_index(drop=True)
    )

    lines = [
        "# 20개 댐 전체 방류량 예측 결과 보고서",
        "",
        "## 1. 목적",
        "",
        "20개 다목적댐 전체에 대해 댐별 개별 방류량 예측 모델을 학습했다.",
        "이 결과는 이후 지도 기반 웹 대시보드에서 각 댐별 예측값을 표시하기 위한 예측 엔진의 기초 자료로 사용한다.",
        "",
        "## 2. 실험 방식",
        "",
        "전체 댐을 하나의 모델로 묶지 않고, 각 댐마다 별도의 모델을 학습했다.",
        "",
        "```text",
        "댐별 데이터 분리",
        "→ 유입량 예측 모델 학습",
        "→ 예측 유입량 생성",
        "→ 방류량 예측 모델 학습",
        "→ 전체 기간 및 이벤트 구간 평가",
        "```",
        "",
        "## 3. 전체 기간 기준 댐별 최고 모델",
        "",
        "| 댐 | 입력변수세트 | 방류량모델 | 기준선 MAE | 모델 MAE | 개선율(%) | R2 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in best_df.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량모델']} | "
            f"{row['전체_기준선_MAE']:.3f} | {row['전체_모델_MAE']:.3f} | "
            f"{row['전체_개선율(%)']:.2f} | {row['전체_모델_R2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 방류량 변화 구간 기준 댐별 최고 모델",
            "",
            "방류량은 평상시 유지되는 구간이 많기 때문에, 실제 방류량이 변한 구간을 별도로 평가했다.",
            "",
            "| 댐 | 입력변수세트 | 방류량모델 | 평가건수 | 기준선 MAE | 모델 MAE | 개선율(%) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in event_best.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량모델']} | "
            f"{int(row['평가건수'])} | {row['기준선_MAE']:.3f} | "
            f"{row['모델_MAE']:.3f} | {row['개선율(%)']:.2f} |"
        )

    improved_overall = (best_df["전체_개선율(%)"] > 0).sum()
    improved_event = (event_best["개선율(%)"] > 0).sum()
    lines.extend(
        [
            "",
            "## 5. 요약",
            "",
            f"- 전체 기간 기준으로 기준선을 이긴 댐: {improved_overall}개 / 20개",
            f"- 방류량 변화 구간 기준으로 기준선을 이긴 댐: {improved_event}개 / 20개",
            "",
            "웹 대시보드에서는 전체 기간 성능만 표시하지 않고, 방류량 변화 구간 성능도 함께 보여주는 것이 적절하다.",
            "실시간 예측 화면에서는 댐별 예측값, 기준선 대비 차이, 위험 또는 변화 가능성 상태를 지도 위에 표시하는 방향이 좋다.",
        ]
    )

    (OUTPUT_DIR / "20개댐_방류량예측_전체보고서.md").write_text("\n".join(lines), encoding="utf-8")
    event_best.to_csv(OUTPUT_DIR / "20개댐_방류량예측_방류량변화구간_댐별최고모델.csv", index=False, encoding="utf-8-sig")


def main():
    base = load_base_module()
    raw = pd.read_csv(base.DATA_PATH, parse_dates=["obsrdt"])
    grade = pd.read_csv(base.GRADE_PATH)
    dams = grade[["댐코드", "댐이름", "성능등급"]].sort_values("댐코드")

    all_overall = []
    all_final = []
    all_predictions = []
    all_inflow_rankings = []

    for row in dams.itertuples(index=False):
        print(f"{row.댐이름} 방류량 예측 전체 확장 학습 중...", flush=True)
        overall, final_eval, predictions, inflow_ranking = base.train_one(
            raw, row.댐코드, row.댐이름, row.성능등급
        )
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
    best_predictions = prediction_df.merge(
        best_keys, on=["댐코드", "입력변수세트", "방류량모델"], how="inner"
    )

    overall_df.to_csv(OUTPUT_DIR / "20개댐_방류량예측_전체평가결과.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUTPUT_DIR / "20개댐_방류량예측_댐별최고모델.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(OUTPUT_DIR / "20개댐_방류량예측_새평가기준결과.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(OUTPUT_DIR / "20개댐_방류량예측_최고모델예측결과.csv", index=False, encoding="utf-8-sig")
    inflow_ranking_df.to_csv(OUTPUT_DIR / "20개댐_방류량예측용_유입량모델평가.csv", index=False, encoding="utf-8-sig")
    write_report(best_df, final_df)

    print("\n20개 댐 전체 기간 기준 최고 모델")
    print(
        best_df[
            [
                "댐이름",
                "유입량성능등급",
                "입력변수세트",
                "방류량모델",
                "전체_기준선_MAE",
                "전체_모델_MAE",
                "전체_개선율(%)",
                "전체_모델_R2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
