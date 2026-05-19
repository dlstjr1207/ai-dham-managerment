from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error


OUTPUT_DIR = Path("우수댐 방류량 예측")
ALL_RESULT_PATH = OUTPUT_DIR / "우수댐_방류량예측_전체평가결과.csv"
BEST_RESULT_PATH = OUTPUT_DIR / "우수댐_방류량예측_댐별최고모델.csv"
BEST_PREDICTION_PATH = OUTPUT_DIR / "우수댐_방류량예측_최고모델예측결과.csv"
EVENT_BEST_PATH = OUTPUT_DIR / "우수댐_방류량예측_최고모델_이벤트평가.csv"
REPORT_PATH = OUTPUT_DIR / "우수댐_방류량예측_심층분석보고서.md"
EVENT_COMPARE_PATH = OUTPUT_DIR / "우수댐_방류량예측_최고모델_이벤트기준선비교.csv"
FEATURE_COMPARE_PATH = OUTPUT_DIR / "우수댐_방류량예측_입력변수세트비교.csv"


def make_feature_compare(all_results):
    deploy_order = {"기본변수": 0, "예측유입량포함": 1, "실제미래유입량_참고용": 2}
    best_by_set = (
        all_results.sort_values(["댐코드", "입력변수세트", "모델_MAE"])
        .groupby(["댐코드", "댐이름", "입력변수세트"], as_index=False)
        .first()
    )
    best_by_set["_순서"] = best_by_set["입력변수세트"].map(deploy_order)
    return best_by_set.sort_values(["댐코드", "_순서"]).drop(columns=["_순서"])


def make_event_compare(best_predictions, event_best):
    rows = []
    for _, event in event_best.iterrows():
        subset = best_predictions[
            (best_predictions["댐코드"] == event["댐코드"])
            & (best_predictions["방류량_모델"] == event["방류량_모델"])
            & (best_predictions["입력변수세트"] == event["입력변수세트"])
        ].copy()

        high_threshold = event["고방류_기준값"]
        high_mask = subset["target_discharge_3h"] >= high_threshold
        change_mask = (subset["target_discharge_3h"] - subset["totdcwtrqy"]).abs() >= 5

        def compare(mask):
            if mask.sum() == 0:
                return pd.Series(
                    {
                        "기준선_MAE": None,
                        "모델_MAE": None,
                        "개선율(%)": None,
                    }
                )
            baseline_mae = mean_absolute_error(
                subset.loc[mask, "target_discharge_3h"],
                subset.loc[mask, "totdcwtrqy"],
            )
            model_mae = mean_absolute_error(
                subset.loc[mask, "target_discharge_3h"],
                subset.loc[mask, "예측방류량"],
            )
            improvement = (
                (baseline_mae - model_mae) / baseline_mae * 100
                if baseline_mae != 0
                else 0
            )
            return pd.Series(
                {
                    "기준선_MAE": baseline_mae,
                    "모델_MAE": model_mae,
                    "개선율(%)": improvement,
                }
            )

        high = compare(high_mask)
        change = compare(change_mask)
        rows.append(
            {
                "댐코드": event["댐코드"],
                "댐이름": event["댐이름"],
                "입력변수세트": event["입력변수세트"],
                "방류량_모델": event["방류량_모델"],
                "고방류_기준값": high_threshold,
                "고방류_건수": int(high_mask.sum()),
                "고방류_기준선_MAE": high["기준선_MAE"],
                "고방류_모델_MAE": high["모델_MAE"],
                "고방류_개선율(%)": high["개선율(%)"],
                "방류변화_건수": int(change_mask.sum()),
                "방류변화_기준선_MAE": change["기준선_MAE"],
                "방류변화_모델_MAE": change["모델_MAE"],
                "방류변화_개선율(%)": change["개선율(%)"],
            }
        )
    return pd.DataFrame(rows)


def verdict(row):
    improvement = row["MAE_개선율(%)"]
    r2 = row["모델_R2"]
    if improvement > 20:
        return "방류량 예측 적용 후보"
    if improvement > 0:
        return "제한적 적용 후보"
    if r2 > 0.85:
        return "패턴은 잡지만 기준선보다 평균오차가 큼"
    return "현재 방식 재검토 필요"


def write_report(best, feature_compare, event_compare):
    lines = [
        "# 우수 등급 댐 방류량 예측 심층분석 보고서",
        "",
        "## 1. 핵심 결론",
        "",
        "유입량 예측에서 우수 등급이었던 4개 댐을 대상으로 방류량 예측을 수행한 결과, 방류량은 유입량보다 훨씬 어렵게 나타났다.",
        "합천댐은 기준선보다 MAE가 뚜렷하게 개선되었지만, 남강댐, 섬진강댐, 용담댐은 R2가 높아도 MAE 기준으로는 현재 방류량 유지 기준선보다 나빴다.",
        "",
        "즉, 방류량은 강수량과 유입량만의 자연 반응이 아니라 운영 판단이 크게 반영되는 값으로 해석해야 한다.",
        "",
        "## 2. 댐별 최종 실사용 후보",
        "",
        "| 댐 | 입력변수세트 | 모델 | 유입량예측모델 | 기준선 MAE | 모델 MAE | 개선율(%) | R2 | 판단 |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]

    for _, row in best.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량_모델']} | "
            f"{row['유입량예측모델']} | {row['기준선_MAE']:.3f} | {row['모델_MAE']:.3f} | "
            f"{row['MAE_개선율(%)']:.2f} | {row['모델_R2']:.3f} | {verdict(row)} |"
        )

    lines.extend(
        [
            "",
            "## 3. 입력변수세트별 비교",
            "",
            "실사용 가능한 세트는 `기본변수`와 `예측유입량포함`이다.",
            "`실제미래유입량_참고용`은 예측 시점에 알 수 없는 값을 사용하므로 최종 모델로 쓸 수 없고, 상한선 확인용이다.",
            "",
            "| 댐 | 입력변수세트 | 최고모델 | 실사용 | 모델 MAE | 개선율(%) | R2 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )

    for _, row in feature_compare.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량_모델']} | "
            f"{row['실사용가능여부']} | {row['모델_MAE']:.3f} | "
            f"{row['MAE_개선율(%)']:.2f} | {row['모델_R2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 고방류 및 방류변화 구간 비교",
            "",
            "평상시에는 방류량이 거의 유지되는 경우가 많기 때문에 현재 방류량 유지 기준선이 매우 강하다.",
            "따라서 고방류 구간과 방류량이 5 이상 변한 구간을 따로 비교했다.",
            "",
            "| 댐 | 고방류 기준 | 고방류 건수 | 고방류 기준선 MAE | 고방류 모델 MAE | 고방류 개선율(%) | 방류변화 건수 | 방류변화 기준선 MAE | 방류변화 모델 MAE | 방류변화 개선율(%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in event_compare.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['고방류_기준값']:.3f} | {int(row['고방류_건수'])} | "
            f"{row['고방류_기준선_MAE']:.3f} | {row['고방류_모델_MAE']:.3f} | "
            f"{row['고방류_개선율(%)']:.2f} | {int(row['방류변화_건수'])} | "
            f"{row['방류변화_기준선_MAE']:.3f} | {row['방류변화_모델_MAE']:.3f} | "
            f"{row['방류변화_개선율(%)']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 댐별 해석",
            "",
        ]
    )

    for _, row in best.iterrows():
        name = row["댐이름"]
        improvement = row["MAE_개선율(%)"]
        r2 = row["모델_R2"]
        if name == "합천댐":
            text = "기준선 대비 MAE가 크게 줄어 현재 방식으로 방류량 예측을 이어갈 가치가 가장 크다."
        elif improvement < 0 and r2 > 0.85:
            text = "전체적인 변동 방향은 어느 정도 따라가지만, 현재 방류량을 그대로 유지하는 기준선보다 평균 오차가 커 실사용 모델로는 아직 부족하다."
        else:
            text = "현재 변수 조합에서는 기준선 대비 개선이 부족하다. 방류량 급변 이벤트 중심 분석이나 운영 규칙 변수를 추가해야 한다."
        lines.append(f"- {name}: {text} 개선율 {improvement:.2f}%, R2 {r2:.3f}.")

    lines.extend(
        [
            "",
            "## 6. 다음 작업 제안",
            "",
            "1. 합천댐을 1순위로 실제값-예측값 그래프와 이벤트 구간 상세 분석을 만든다.",
            "2. 남강댐, 섬진강댐, 용담댐은 기준선이 강한 이유를 확인한다. 방류량이 장시간 유지되는 구간이 많으면 기준선이 쉽게 이긴다.",
            "3. 방류량 모델에는 운영 판단을 대체할 수 있는 변수가 필요하다. 예를 들면 저수율 구간, 홍수기 여부, 월/계절, 방류량 변화 여부 분류 변수를 추가할 수 있다.",
            "4. 바로 20개 댐 전체 방류량으로 확장하기보다, 합천댐에서 모델 구조를 먼저 고도화한 뒤 확장하는 것이 안전하다.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    all_results = pd.read_csv(ALL_RESULT_PATH)
    best = pd.read_csv(BEST_RESULT_PATH)
    best_predictions = pd.read_csv(BEST_PREDICTION_PATH)
    event_best = pd.read_csv(EVENT_BEST_PATH)

    feature_compare = make_feature_compare(all_results)
    event_compare = make_event_compare(best_predictions, event_best)

    feature_compare.to_csv(FEATURE_COMPARE_PATH, index=False, encoding="utf-8-sig")
    event_compare.to_csv(EVENT_COMPARE_PATH, index=False, encoding="utf-8-sig")
    write_report(best, feature_compare, event_compare)

    print(best[["댐이름", "입력변수세트", "방류량_모델", "기준선_MAE", "모델_MAE", "MAE_개선율(%)", "모델_R2"]].to_string(index=False))
    print()
    print(event_compare[["댐이름", "고방류_개선율(%)", "방류변화_개선율(%)"]].to_string(index=False))


if __name__ == "__main__":
    main()
