from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
API_ROOT = BASE_DIR.parent
PROJECT_ROOT = API_ROOT.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PIPELINE_STEPS = [
    ("수문 운영 정보 수집", API_ROOT / "01_API수집" / "api_수문운영정보_수집.py", ["--hours", "72"]),
    ("기상청 초단기 수집", API_ROOT / "01_API수집" / "api_기상청_초단기_수집.py", ["--mode", "both"]),
    ("기상청 단기예보 수집", API_ROOT / "01_API수집" / "api_기상청_단기예보_수집.py", []),
    ("실시간 모델 입력 생성", API_ROOT / "02_실시간입력" / "실시간_모델입력데이터_생성.py", []),
    ("실시간 ML 예측", API_ROOT / "03_실시간예측" / "실시간_ML예측엔진.py", []),
]


def load_env_for_sanitize() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().lstrip("\ufeff"), value.strip().strip('"').strip("'"))


def sanitize(text: str) -> str:
    load_env_for_sanitize()
    service_key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
    redacted = text
    if service_key:
        redacted = redacted.replace(service_key, "[SERVICE_KEY_REDACTED]")
    redacted = re.sub(r"(serviceKey=)[^&\s]+", r"\1[SERVICE_KEY_REDACTED]", redacted)
    return redacted


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {sanitize(message)}"
    print(line, flush=True)
    with (LOG_DIR / "자동수집.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, script_path: Path, extra_args: list[str]) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"{name} 스크립트를 찾을 수 없습니다: {script_path}")

    command = [sys.executable, str(script_path), *extra_args]
    log(f"{name} 시작")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.stdout.strip():
        log(f"{name} 출력\n{completed.stdout.strip()}")
    if completed.stderr.strip():
        log(f"{name} 경고/오류 출력\n{completed.stderr.strip()}")
    if completed.returncode != 0:
        raise RuntimeError(f"{name} 실패: 종료 코드 {completed.returncode}")
    log(f"{name} 완료")


def run_pipeline() -> bool:
    started_at = datetime.now()
    log(f"실시간 API 자동수집 파이프라인 시작: {started_at:%Y-%m-%d %H:%M:%S}")
    try:
        for name, script_path, extra_args in PIPELINE_STEPS:
            run_step(name, script_path, extra_args)
    except Exception as exc:
        log(f"파이프라인 실패: {exc}")
        return False

    finished_at = datetime.now()
    elapsed = (finished_at - started_at).total_seconds()
    log(f"실시간 API 자동수집 파이프라인 완료: {finished_at:%Y-%m-%d %H:%M:%S}, 소요 {elapsed:.1f}초")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="30분 단위 실시간 API 수집/예측 자동 실행")
    parser.add_argument("--once", action="store_true", help="한 번만 실행하고 종료")
    parser.add_argument("--interval-minutes", type=int, default=30, help="반복 실행 간격. 기본 30분")
    args = parser.parse_args()

    if args.once:
        success = run_pipeline()
        raise SystemExit(0 if success else 1)

    interval_seconds = max(args.interval_minutes, 1) * 60
    log(f"자동수집 반복 실행 시작: {args.interval_minutes}분 간격")
    while True:
        run_pipeline()
        log(f"다음 실행까지 대기: {args.interval_minutes}분")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
