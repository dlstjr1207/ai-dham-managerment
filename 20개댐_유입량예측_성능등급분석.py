from pathlib import Path

import pandas as pd


RESULT_DIR = Path("20개 댐 유입량 예측")
BEST_MODEL_PATH = RESULT_DIR / "20개댐_개별_유입량예측_댐별최고모델.csv"
GRADE_TABLE_PATH = RESULT_DIR / "20개댐_유입량예측_성능등급표.csv"
REPORT_PATH = RESULT_DIR / "20개댐_유입량예측_성능등급_분석보고서.md"


def classify(row):
    improvement = row["MAE_개선율(%)"]
    r2 = row["모델_R2"]

    if improvement < 0:
        return "부진"
    if improvement >= 14 and r2 >= 0.7:
        return "우수"
    if (improvement > 0 and r2 >= 0.5) or improvement >= 10:
        return "양호"
    return "주의"


def explain(row):
    improvement = row["MAE_개선율(%)"]
    r2 = row["모델_R2"]

    if improvement < 0:
        return "모델 MAE가 단순 기준선보다 커서 현재 변수와 모델로는 개선 효과가 확인되지 않음"
    if improvement >= 14 and r2 >= 0.7:
        return "기준선 대비 오차가 충분히 줄고 R2도 높아 유입량 패턴 설명력이 좋음"
    if improvement >= 10 and r2 < 0.5:
        return "MAE는 줄었지만 R2가 낮아 급변 패턴 설명력은 추가 확인 필요"
    if improvement > 0 and r2 >= 0.5:
        return "기준선 대비 개선은 확인되며 패턴 설명력도 보통 이상임"
    return "개선율이 작거나 R2가 낮아 추가 변수, 모델, 이벤트 중심 분석 필요"


def build_report(analysis):
    lines = [
        "# 20개 댐 유입량 예측 성능 등급 분석 보고서",
        "",
        "## 1. 분석 목적",
        "",
        "20개 다목적댐을 각각 개별 모델로 학습한 뒤, 댐별 유입량 예측 성능을 프로젝트에서 해석 가능한 등급으로 분류했다.",
        "이번 분류는 방류량 예측으로 넘어가기 전에 어떤 댐부터 우선 적용할지 정하기 위한 중간 분석이다.",
        "",
        "## 2. 분류 기준",
        "",
        "| 등급 | 기준 | 해석 |",
        "|---|---|---|",
        "| 우수 | MAE 개선율 14% 이상, R2 0.7 이상 | 기준선보다 오차가 충분히 줄고 패턴 설명력도 좋음 |",
        "| 양호 | MAE 개선율 양수이고 R2 0.5 이상, 또는 개선율 10% 이상 | 실제 적용 후보로 볼 수 있으나 일부 보완 필요 |",
        "| 주의 | 개선율은 양수지만 개선 폭이 작거나 R2가 낮음 | 추가 변수 또는 이벤트 중심 검토 필요 |",
        "| 부진 | MAE 개선율 음수 | 단순 기준선보다 나빠 현재 방식으로는 부적합 |",
        "",
        "기준선은 현재 유입량이 3시간 뒤에도 그대로 유지된다고 가정한 값이다.",
        "",
        "```text",
        "baseline_prediction(t+3) = inflowqy(t)",
        "```",
        "",
        "## 3. 전체 등급표",
        "",
        "| 등급 | 댐 | 최고모델 | 기준선 MAE | 모델 MAE | 개선율(%) | R2 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for _, row in analysis.iterrows():
        lines.append(
            f"| {row['성능등급']} | {row['댐이름']} | {row['모델']} | "
            f"{row['기준선_MAE']:.3f} | {row['모델_MAE']:.3f} | "
            f"{row['MAE_개선율(%)']:.2f} | {row['모델_R2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 등급별 해석",
            "",
        ]
    )

    descriptions = {
        "우수": "방류량 예측으로 우선 연결하기 좋은 댐이다. 유입량 예측이 기준선보다 명확히 좋고 R2도 높아, 강수량과 과거 유입량 기반 패턴이 비교적 잘 잡힌 것으로 볼 수 있다.",
        "양호": "방류량 예측 후보로 사용할 수 있지만, 댐별로 개선율과 R2의 균형을 확인해야 한다. 예측 성능은 확보됐지만 일부 급변 구간에서는 오차가 커질 수 있다.",
        "주의": "평균 오차는 약간 줄었지만 모델 설명력이 충분하다고 보기 어렵다. 강수량 외 추가 변수나 장마철/급증 이벤트 중심 분석이 필요하다.",
        "부진": "현재 입력 변수와 모델 조합에서는 단순 기준선보다 좋지 않았다. 바로 방류량 예측에 넣기보다 데이터 품질, 이상치, 지연 시간, 모델 구조를 먼저 점검해야 한다.",
    }

    for grade in ["우수", "양호", "주의", "부진"]:
        subset = analysis[analysis["성능등급"] == grade]
        dams = ", ".join(subset["댐이름"].tolist()) if not subset.empty else "없음"
        lines.extend(
            [
                f"### {grade}",
                "",
                f"대상 댐: {dams}",
                "",
                descriptions[grade],
                "",
            ]
        )

    lines.extend(
        [
            "## 5. 다음 단계 제안",
            "",
            "방류량 예측은 유입량 예측 성능이 좋은 댐부터 진행하는 것이 적절하다.",
            "우선 대상은 우수 등급 댐이며, 이후 양호 등급 댐까지 확장하는 순서가 좋다.",
            "부진 등급 댐은 바로 방류량 예측에 넣기보다 원인 분석을 먼저 수행해야 한다.",
            "",
            "추천 진행 순서:",
            "",
            "1. 우수 등급 댐 방류량 예측 실험",
            "2. 양호 등급 댐 방류량 예측 확장",
            "3. 주의/부진 등급 댐 원인 분석",
            "4. 최종 발표용 그래프와 표 작성",
        ]
    )

    return "\n".join(lines)


def main():
    best = pd.read_csv(BEST_MODEL_PATH)
    analysis = best.copy()
    analysis["성능등급"] = analysis.apply(classify, axis=1)
    analysis["해석"] = analysis.apply(explain, axis=1)

    grade_order = {"우수": 0, "양호": 1, "주의": 2, "부진": 3}
    analysis["_등급순서"] = analysis["성능등급"].map(grade_order)
    analysis = (
        analysis.sort_values(["_등급순서", "MAE_개선율(%)"], ascending=[True, False])
        .drop(columns=["_등급순서"])
        .reset_index(drop=True)
    )

    analysis.to_csv(GRADE_TABLE_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(analysis), encoding="utf-8")

    print(analysis[["성능등급", "댐이름", "모델", "기준선_MAE", "모델_MAE", "MAE_개선율(%)", "모델_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
