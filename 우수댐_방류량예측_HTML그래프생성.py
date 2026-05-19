from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("우수댐 방류량 예측")
PREDICTION_PATH = OUTPUT_DIR / "우수댐_방류량예측_최고모델예측결과.csv"
HTML_PATH = OUTPUT_DIR / "우수댐_방류량예측_그래프.html"


WIDTH = 980
HEIGHT = 320
PADDING_LEFT = 56
PADDING_RIGHT = 18
PADDING_TOP = 24
PADDING_BOTTOM = 36


def scale_points(df, y_col):
    n = len(df)
    plot_width = WIDTH - PADDING_LEFT - PADDING_RIGHT
    plot_height = HEIGHT - PADDING_TOP - PADDING_BOTTOM
    max_y = max(
        df["target_discharge_3h"].max(),
        df["예측방류량"].max(),
        df["totdcwtrqy"].max(),
        1,
    )
    points = []
    for i, value in enumerate(df[y_col]):
        x = PADDING_LEFT + (i / max(n - 1, 1)) * plot_width
        y = PADDING_TOP + (1 - value / max_y) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points), max_y


def make_line_chart(df, title):
    actual_points, max_y = scale_points(df, "target_discharge_3h")
    pred_points, _ = scale_points(df, "예측방류량")
    baseline_points, _ = scale_points(df, "totdcwtrqy")
    y0 = HEIGHT - PADDING_BOTTOM
    x0 = PADDING_LEFT
    x1 = WIDTH - PADDING_RIGHT
    y_top = PADDING_TOP
    mid_y = (y0 + y_top) / 2

    return f"""
    <section class="chart-card">
      <h2>{title}</h2>
      <svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{title}">
        <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis" />
        <line x1="{x0}" y1="{y_top}" x2="{x0}" y2="{y0}" class="axis" />
        <line x1="{x0}" y1="{mid_y}" x2="{x1}" y2="{mid_y}" class="grid" />
        <text x="8" y="{y_top + 4}" class="tick">{max_y:.1f}</text>
        <text x="18" y="{mid_y + 4}" class="tick">{max_y / 2:.1f}</text>
        <text x="38" y="{y0 + 4}" class="tick">0</text>
        <polyline points="{baseline_points}" class="baseline" />
        <polyline points="{actual_points}" class="actual" />
        <polyline points="{pred_points}" class="pred" />
      </svg>
      <div class="legend">
        <span><i class="actual-box"></i>실제 3시간 뒤 방류량</span>
        <span><i class="pred-box"></i>예측 방류량</span>
        <span><i class="base-box"></i>현재 방류량 기준선</span>
      </div>
    </section>
    """


def peak_window(df):
    peak_time = df.loc[df["target_discharge_3h"].idxmax(), "obsrdt"]
    start = peak_time - pd.Timedelta(days=7)
    end = peak_time + pd.Timedelta(days=7)
    return df[(df["obsrdt"] >= start) & (df["obsrdt"] <= end)].copy()


def make_html(data):
    sections = []
    for dam_name, df in data.groupby("댐이름"):
        df = df.sort_values("obsrdt").reset_index(drop=True)
        sample = df.iloc[:: max(len(df) // 900, 1)].copy()
        sections.append(make_line_chart(sample, f"{dam_name} 전체 평가기간"))
        sections.append(make_line_chart(peak_window(df), f"{dam_name} 최대 방류 이벤트 전후 14일"))

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>우수댐 방류량 예측 그래프</title>
  <style>
    body {{
      margin: 0;
      padding: 28px;
      font-family: "Malgun Gothic", Arial, sans-serif;
      background: #f6f7f8;
      color: #1f2933;
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 26px;
    }}
    .chart-card {{
      background: white;
      border: 1px solid #d9dee3;
      border-radius: 8px;
      padding: 18px 18px 14px;
      margin: 0 0 18px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 17px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .axis {{
      stroke: #8a96a3;
      stroke-width: 1;
    }}
    .grid {{
      stroke: #e4e8ec;
      stroke-width: 1;
    }}
    .actual {{
      fill: none;
      stroke: #0f766e;
      stroke-width: 2.1;
    }}
    .pred {{
      fill: none;
      stroke: #b42318;
      stroke-width: 1.8;
      opacity: .92;
    }}
    .baseline {{
      fill: none;
      stroke: #52616f;
      stroke-width: 1.2;
      stroke-dasharray: 5 4;
      opacity: .75;
    }}
    .tick {{
      fill: #697586;
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-size: 13px;
      color: #344054;
      padding-left: 56px;
    }}
    .legend i {{
      display: inline-block;
      width: 14px;
      height: 3px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .actual-box {{ background: #0f766e; }}
    .pred-box {{ background: #b42318; }}
    .base-box {{ background: #52616f; }}
  </style>
</head>
<body>
  <h1>우수댐 방류량 예측 그래프</h1>
  {''.join(sections)}
</body>
</html>
"""


def main():
    data = pd.read_csv(PREDICTION_PATH, parse_dates=["obsrdt"])
    HTML_PATH.write_text(make_html(data), encoding="utf-8")
    print(f"그래프 HTML 생성 완료: {HTML_PATH}")


if __name__ == "__main__":
    main()
