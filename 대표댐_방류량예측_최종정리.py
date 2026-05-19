from pathlib import Path

import pandas as pd


CANDIDATE_DIR = Path("후보댐 방류량 예측")
OUTPUT_DIR = Path("대표댐 방류량 예측 최종정리")
OUTPUT_DIR.mkdir(exist_ok=True)

OVERALL_PATH = CANDIDATE_DIR / "후보댐_방류량예측_전체평가결과.csv"
FINAL_STANDARD_PATH = CANDIDATE_DIR / "후보댐_방류량예측_새평가기준결과.csv"
PREDICTION_PATH = CANDIDATE_DIR / "후보댐_방류량예측_최고모델예측결과.csv"

REPRESENTATIVE_DAMS = ["합천댐", "주암(조)댐", "대청댐", "임하댐", "남강댐"]


def select_event_models(final_eval):
    event = final_eval[
        (final_eval["평가구간"] == "방류량변화")
        & (final_eval["기준선"] == "현재방류량유지")
        & (final_eval["댐이름"].isin(REPRESENTATIVE_DAMS))
    ].copy()
    idx = event.groupby("댐코드")["개선율(%)"].idxmax()
    return (
        event.loc[idx]
        .sort_values("개선율(%)", ascending=False)
        .reset_index(drop=True)
    )


def select_overall_models(overall):
    target = overall[overall["댐이름"].isin(REPRESENTATIVE_DAMS)].copy()
    idx = target.groupby("댐코드")["전체_모델_MAE"].idxmin()
    return (
        target.loc[idx]
        .sort_values("전체_개선율(%)", ascending=False)
        .reset_index(drop=True)
    )


def build_event_summary(event_models, final_eval):
    segments = ["전체", "최근24시간누적강우", "유입량급증", "방류량변화"]
    rows = []
    for _, model in event_models.iterrows():
        subset = final_eval[
            (final_eval["댐코드"] == model["댐코드"])
            & (final_eval["입력변수세트"] == model["입력변수세트"])
            & (final_eval["방류량모델"] == model["방류량모델"])
            & (final_eval["기준선"] == "현재방류량유지")
            & (final_eval["평가구간"].isin(segments))
        ].copy()
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def create_html_graph(prediction):
    graph_path = OUTPUT_DIR / "대표댐_방류량예측_그래프.html"
    width = 980
    height = 310
    left = 58
    right = 18
    top = 24
    bottom = 36

    def points(df, column):
        max_y = max(df["target_discharge_3h"].max(), df["예측방류량"].max(), df["totdcwtrqy"].max(), 1)
        plot_w = width - left - right
        plot_h = height - top - bottom
        out = []
        for i, value in enumerate(df[column]):
            x = left + (i / max(len(df) - 1, 1)) * plot_w
            y = top + (1 - value / max_y) * plot_h
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out), max_y

    sections = []
    for dam_name, df in prediction[prediction["댐이름"].isin(REPRESENTATIVE_DAMS)].groupby("댐이름"):
        df = df.sort_values("obsrdt").reset_index(drop=True)
        sample = df.iloc[:: max(len(df) // 900, 1)].copy()
        actual, max_y = points(sample, "target_discharge_3h")
        pred, _ = points(sample, "예측방류량")
        base, _ = points(sample, "totdcwtrqy")
        y0 = height - bottom
        mid = (top + y0) / 2
        sections.append(
            f"""
            <section>
              <h2>{dam_name}</h2>
              <svg viewBox="0 0 {width} {height}">
                <line x1="{left}" y1="{y0}" x2="{width-right}" y2="{y0}" class="axis" />
                <line x1="{left}" y1="{top}" x2="{left}" y2="{y0}" class="axis" />
                <line x1="{left}" y1="{mid}" x2="{width-right}" y2="{mid}" class="grid" />
                <text x="8" y="{top+4}" class="tick">{max_y:.1f}</text>
                <text x="18" y="{mid+4}" class="tick">{max_y/2:.1f}</text>
                <text x="40" y="{y0+4}" class="tick">0</text>
                <polyline points="{base}" class="base" />
                <polyline points="{actual}" class="actual" />
                <polyline points="{pred}" class="pred" />
              </svg>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>대표댐 방류량 예측 그래프</title>
  <style>
    body {{ margin: 0; padding: 28px; font-family: "Malgun Gothic", Arial, sans-serif; background: #f5f6f7; color: #1f2933; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    p {{ margin: 0 0 20px; color: #52616f; }}
    section {{ background: #fff; border: 1px solid #d9dee3; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    h2 {{ margin: 0 0 8px; font-size: 17px; }}
    svg {{ display: block; width: 100%; height: auto; }}
    .axis {{ stroke: #8a96a3; stroke-width: 1; }}
    .grid {{ stroke: #e4e8ec; stroke-width: 1; }}
    .actual {{ fill: none; stroke: #0f766e; stroke-width: 2.1; }}
    .pred {{ fill: none; stroke: #b42318; stroke-width: 1.8; opacity: .9; }}
    .base {{ fill: none; stroke: #52616f; stroke-width: 1.1; stroke-dasharray: 5 4; opacity: .72; }}
    .tick {{ fill: #697586; font-size: 12px; }}
    .legend {{ display: flex; gap: 18px; margin-bottom: 18px; font-size: 13px; }}
    .legend i {{ display: inline-block; width: 16px; height: 3px; margin-right: 6px; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>대표댐 방류량 예측 그래프</h1>
  <p>실선 초록: 실제 3시간 뒤 방류량, 빨강: 예측 방류량, 회색 점선: 현재 방류량 유지 기준선</p>
  <div class="legend">
    <span><i style="background:#0f766e"></i>실제</span>
    <span><i style="background:#b42318"></i>예측</span>
    <span><i style="background:#52616f"></i>기준선</span>
  </div>
  {''.join(sections)}
</body>
</html>
"""
    graph_path.write_text(html, encoding="utf-8")


def write_report(overall_models, event_models, event_summary):
    lines = [
        "# 대표댐 방류량 예측 최종정리 보고서",
        "",
        "## 1. 대표댐 선정",
        "",
        "새 평가 기준을 적용한 결과, 방류량 변화 구간에서 개선 효과가 큰 댐을 대표 분석 대상으로 선정했다.",
        "",
        "대표댐은 합천댐, 주암(조)댐, 대청댐, 임하댐, 남강댐이다.",
        "",
        "## 2. 전체 기간 기준 모델",
        "",
        "| 댐 | 입력변수세트 | 모델 | 기준선 MAE | 모델 MAE | 개선율(%) | R2 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in overall_models.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량모델']} | "
            f"{row['전체_기준선_MAE']:.3f} | {row['전체_모델_MAE']:.3f} | "
            f"{row['전체_개선율(%)']:.2f} | {row['전체_모델_R2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 3. 방류량 변화 구간 기준 모델",
            "",
            "프로젝트 목적상 실제 방류량이 변하는 구간을 별도로 평가하는 것이 중요하다.",
            "아래 표는 방류량 변화 구간에서 가장 좋은 모델을 댐별로 고른 결과이다.",
            "",
            "| 댐 | 입력변수세트 | 모델 | 평가건수 | 기준선 MAE | 모델 MAE | 개선율(%) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in event_models.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['입력변수세트']} | {row['방류량모델']} | "
            f"{int(row['평가건수'])} | {row['기준선_MAE']:.3f} | {row['모델_MAE']:.3f} | "
            f"{row['개선율(%)']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 대표댐별 구간 평가",
            "",
            "| 댐 | 평가구간 | 평가건수 | 기준선 MAE | 모델 MAE | 개선율(%) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in event_summary.iterrows():
        lines.append(
            f"| {row['댐이름']} | {row['평가구간']} | {int(row['평가건수'])} | "
            f"{row['기준선_MAE']:.3f} | {row['모델_MAE']:.3f} | {row['개선율(%)']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 결론",
            "",
            "전체 기간에서는 현재 방류량 유지 기준선이 강해서 모델 성능이 낮게 보일 수 있다.",
            "하지만 방류량 변화 구간에서는 대표댐 대부분에서 모델이 기준선을 크게 이겼다.",
            "",
            "따라서 최종 보고서에서는 전체 기간 성능과 이벤트 구간 성능을 함께 제시해야 한다.",
            "특히 방류량 변화 구간의 개선율은 프로젝트의 핵심 성과로 사용할 수 있다.",
            "",
            "다음 단계는 대표댐 중심으로 발표용 그래프와 최종 요약 슬라이드에 들어갈 표를 정리하는 것이다.",
        ]
    )
    (OUTPUT_DIR / "대표댐_방류량예측_최종정리보고서.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    overall = pd.read_csv(OVERALL_PATH)
    final_eval = pd.read_csv(FINAL_STANDARD_PATH)
    prediction = pd.read_csv(PREDICTION_PATH, parse_dates=["obsrdt"])

    overall_models = select_overall_models(overall)
    event_models = select_event_models(final_eval)
    event_summary = build_event_summary(event_models, final_eval)

    overall_models.to_csv(OUTPUT_DIR / "대표댐_전체기간_최고모델.csv", index=False, encoding="utf-8-sig")
    event_models.to_csv(OUTPUT_DIR / "대표댐_방류량변화구간_최고모델.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(OUTPUT_DIR / "대표댐_구간별_평가요약.csv", index=False, encoding="utf-8-sig")
    create_html_graph(prediction)
    write_report(overall_models, event_models, event_summary)

    print("대표댐 방류량 변화 구간 최고 모델")
    print(event_models[["댐이름", "입력변수세트", "방류량모델", "평가건수", "기준선_MAE", "모델_MAE", "개선율(%)"]].to_string(index=False))


if __name__ == "__main__":
    main()
