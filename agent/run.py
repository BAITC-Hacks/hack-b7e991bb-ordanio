# Суточный цикл агента и прогон по периоду: run_day, run_period, RunResult, командная строка.
# Запуск: python -m agent.run --date 2026-02-05  или  python -m agent.run --from 2026-01-31 --to 2026-02-28

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from common.config import TEST_ISSUE_DATES, TZ
from agent import tools
from agent.analyst import daily_note

log = logging.getLogger("agent")


@dataclass
class RunResult:
    """Итог одного дня: дата выпуска, шаги с временем, путь к CSV, анализ, сводка."""
    issue_date: str
    steps: list = field(default_factory=list)
    forecast_path: str = ""
    analysis: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class _Step:
    """Замер одного шага: время начала и конца, краткий итог для журнала шагов."""

    def __init__(self, result: RunResult, name: str):
        self.result, self.name = result, name
        self.started = _now()
        self.t0 = _time.perf_counter()

    def done(self, summary: str) -> None:
        self.result.steps.append({
            "name": self.name, "started": self.started, "finished": _now(),
            "seconds": round(_time.perf_counter() - self.t0, 3), "summary": summary,
        })
        log.info("Шаг %s готов за %.2f с: %s", self.name, _time.perf_counter() - self.t0, summary)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_day(issue_date: str, previous_forecast: pd.DataFrame | None = None) -> RunResult:
    """Один день: погода → признаки → модель → CSV → анализ → сводка → журнал; пишет output/runs/<date>.json.
    Если вчерашний прогноз не передан, берётся из output/forecasts, когда файл есть."""
    issue_date = validate_issue_date(issue_date)
    result = RunResult(issue_date=issue_date)
    log.info("=== День выпуска %s ===", issue_date)

    step = _Step(result, "fetch_weather")
    weather = tools.fetch_weather(issue_date)
    source = tools._weather_source(weather)
    step.done(f"{len(weather)} часов погоды, источник: {tools._source_ru(source)}, "
              f"ветер на 100 м {weather['ws100'].min():.1f}…{weather['ws100'].max():.1f} м/с")

    step = _Step(result, "prepare")
    features = tools.prepare(weather)
    step.done(f"признаки для {len(features)} турбин по {len(weather)} часам")

    step = _Step(result, "run_model")
    forecast = tools.run_model(features)
    step.done("p10/p50/p90 по часам, сумма p50: " + ", ".join(
        f"турбина {t} {forecast[forecast['turbine'] == t]['p50'].sum():.2f}" for t in features))

    if previous_forecast is None:
        previous_forecast = tools.load_previous_forecast(issue_date)
        if previous_forecast is not None:
            log.info("Вчерашний прогноз взят из файла output/forecasts")

    actuals = tools.load_actuals(issue_date)
    if previous_forecast is None and actuals is not None:
        # Факт за день выпуска есть (31.01), а файла вчерашнего прогноза нет: день D−1 вне периода
        # выпуска, поэтому считаем его прогноз тем же циклом в памяти, без записи файла.
        step = _Step(result, "previous_forecast")
        prev_date = (pd.Timestamp(issue_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prev_weather = tools.fetch_weather(prev_date)
        previous_forecast = tools.run_model(tools.prepare(prev_weather))
        step.done(f"прогноз выпуска {prev_date} пересчитан в памяти для сверки с фактом за {issue_date}")

    step = _Step(result, "save_forecast")
    result.forecast_path = tools.save_forecast(issue_date, forecast, weather, previous_forecast)
    step.done(f"{result.forecast_path}, {len(forecast)} строк")

    step = _Step(result, "analyze")
    analysis = tools.analyze(issue_date, forecast, previous_forecast, actuals)
    analysis["weather_source"] = source
    result.analysis = analysis
    step.done(f"сумма p50 {analysis['totals']['all']['total']:.2f}, низкой уверенности "
              f"{analysis['low_confidence_count']} ч, экстремального ветра {len(analysis['extreme_wind_hours'])} ч")

    step = _Step(result, "note")
    result.note = daily_note(analysis)
    step.done(result.note.splitlines()[0][:120] if result.note else "пусто")

    step = _Step(result, "write_journal")
    tools.write_journal(issue_date, analysis, result.note)
    step.done(tools.JOURNAL_PATH)

    os.makedirs(tools.RUNS_DIR, exist_ok=True)
    run_path = os.path.join(tools.RUNS_DIR, f"{issue_date}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    log.info("Итог дня записан: %s", run_path)
    return result


def run_period(start: str, end: str) -> list[RunResult]:
    """Последовательно по датам; вчерашний прогноз передаётся в следующий день для сравнения."""
    start, end = validate_issue_date(start), validate_issue_date(end)
    if start > end:
        raise ValueError(f"Начало периода {start} позже конца {end}")
    results = []
    previous = None
    for day in pd.date_range(start, end, freq="D"):
        issue_date = day.strftime("%Y-%m-%d")
        result = run_day(issue_date, previous)
        results.append(result)
        previous = tools.read_forecast_csv(result.forecast_path)
    log.info("Период %s … %s: %d дней, файлы в %s, журнал %s",
             start, end, len(results), tools.FORECASTS_DIR, tools.JOURNAL_PATH)
    return results


def validate_issue_date(value: str) -> str:
    """Проверяет формат ГГГГ-ММ-ДД и попадание в тестовый период; иначе ValueError с русским текстом."""
    if value is None or not str(value).strip():
        raise ValueError("Дата выпуска не указана. Формат: ГГГГ-ММ-ДД, например 2026-02-05.")
    text = str(value).strip()
    try:
        day = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Дата «{text}» не в формате ГГГГ-ММ-ДД, например 2026-02-05.") from None
    issue_date = day.strftime("%Y-%m-%d")
    lo, hi = TEST_ISSUE_DATES
    if not (lo <= issue_date <= hi):
        raise ValueError(f"Дата {issue_date} вне тестового периода: допустимы даты с {lo} по {hi}.")
    return issue_date


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.run",
        description="Прогноз выработки двух турбин на 48 часов от даты выпуска (31.01–28.02.2026).",
    )
    parser.add_argument("--date", help="один день выпуска, ГГГГ-ММ-ДД")
    parser.add_argument("--from", dest="start", help="начало периода, ГГГГ-ММ-ДД")
    parser.add_argument("--to", dest="end", help="конец периода, ГГГГ-ММ-ДД")
    args = parser.parse_args(argv)
    _setup_logging()

    if args.date and (args.start or args.end):
        print("Укажите либо --date, либо пару --from и --to, но не всё сразу.")
        return 2
    if not args.date and not (args.start and args.end):
        print("Укажите дату: --date ГГГГ-ММ-ДД, либо период: --from ГГГГ-ММ-ДД --to ГГГГ-ММ-ДД.")
        return 2
    try:  # проверка ввода отдельно: кривая дата — код 2, сбой расчёта — код 1
        if args.date:
            validate_issue_date(args.date)
        else:
            start, end = validate_issue_date(args.start), validate_issue_date(args.end)
            if start > end:
                raise ValueError(f"Начало периода {start} позже конца {end}")
    except ValueError as exc:
        print(f"Ошибка входных данных: {exc}")
        return 2
    try:
        if args.date:
            result = run_day(args.date)
            print(f"Готово: {result.forecast_path}, журнал {tools.JOURNAL_PATH}")
        else:
            results = run_period(args.start, args.end)
            print(f"Готово: {len(results)} дней, файлы в {tools.FORECASTS_DIR}, журнал {tools.JOURNAL_PATH}")
        return 0
    except FileNotFoundError as exc:
        print(f"Не найден файл: {exc}. Проверьте, что данные и кэш погоды лежат в data/, а модель обучена.")
        return 1
    except Exception as exc:  # любой сбой показываем текстом, без трейсбека
        log.debug("Сбой", exc_info=True)
        print(f"Сбой при расчёте: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
