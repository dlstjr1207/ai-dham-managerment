# AI Dam Discharge Event Detection & Decision Support System

수문·기상·하류 관측 데이터를 결합해
댐 방류량의 급격한 변화 가능성을 3시간·6시간 전에 탐지하고,
관리자의 검토 우선순위를 제공하는 머신러닝 기반 의사결정 지원 시스템입니다.

> 자동 방류 제어 시스템이 아니라,
> 방류 변화 가능성이 높은 댐과 시점을 먼저 확인하도록 돕는 연구 프로토타입입니다.

## Key Results
- 20개 다목적댐 대상 시계열 검증
- Two-Stage HistGradientBoosting 기반 3h·6h 예측
- 이벤트 구간 Recall 개선
  - 3h: 0.245 → 0.330
  - 6h: 0.244 → 0.325
- 이벤트 구간 ΔMAE 개선
  - 3h: 20.25 → 23.03 m³/s
  - 6h: 19.45 → 23.13 m³/s

## System Architecture
1. Stage 1: 미래 유입량 예측
2. Stage 2-A: 부호 있는 방류 변화량 예측
3. Stage 2-B: 방류 변화 이벤트 분류
4. Streamlit Dashboard: 관리자 검토 우선순위 시각화

## Event Definition
방류 변화 이벤트는 현재 방류량과 미래 방류량의 변화폭이
댐별 임계값 이상인 경우로 정의했습니다.

|Qout(t+h) - Qout(t)| ≥ Td

## Tech Stack
- Python, Pandas, NumPy
- scikit-learn HistGradientBoosting
- Streamlit
- K-water / 기상청 공공 API
- Plotly

## Dashboard

<img width="1800" height="885" alt="KakaoTalk_20260520_200722128" src="https://github.com/user-attachments/assets/ca1304f2-04a7-4f4e-9f9f-327833b0e676" />

- 현재 수문 상태
- 3h·6h 방류 변화량 및 이벤트 확률
- 최근 72시간 수문 변화
- 관리자 판단 요약

## My Contributions
- 시계열 데이터 전처리 및 누수 방지 검증 설계
- 댐별 이벤트 임계값 산정
- Two-Stage 방류 변화 예측·분류 파이프라인 구현
- 하류 수위·유량 보조 피처 연동
- Streamlit 대시보드 설계 및 구현
- 발표 자료·논문 작성

## Limitations
- 방류 승인 이력, 발전·용수 공급 계획 등 실제 운영 제약 변수는 포함하지 못했습니다.
- 실시간 자동 제어가 아닌 연구용 판단 보조 프로토타입입니다.
