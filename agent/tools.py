# Инструменты агента: каждый шаг суточного цикла как чистая функция с логом входа и выхода.
# Здесь же правила анализа (сумма энергии, сравнение с вчерашним прогнозом, низкая уверенность) и журнал.

from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from common.config import ARTIFACTS, HORIZON_HOURS, OUTPUT, TURBINES, TZ
from model.features import build_features
from model.predict import predict
from model.weather import get_issued_forecast

log = logging.getLogger("agent")

FORECAST_COLUMNS = [
    "issue_date", "target_time", "lead_hours", "issue_timestamp", "weather_lead_hours", "weather_run_time",
    "turbine", "ws100_forecast", "temp_forecast", "p10", "p50", "p90", "confidence", "note",
]
ISSUE_TIME = "23:59"       # момент выпуска прогноза: конец дня D по местному времени
FORECASTS_DIR = os.path.join(OUTPUT, "forecasts")
RUNS_DIR = os.path.join(OUTPUT, "runs")
JOURNAL_PATH = os.path.join(OUTPUT, "journal.md")

CUTOUT_WS100 = 25.0        # м/с: гипотеза штатной остановки, паспортный порог турбин неизвестен
DELTA_P50_LOW = 0.15       # порог изменения p50 к вчерашнему прогнозу
WIDTH_LOW_FALLBACK = 0.8   # запасной порог ширины коридора p90 − p10, если январской проверки модели нет
WIDTH_QUANTILE = 0.75      # порог ширины: 75-й процентиль ширины по validation.csv, верхняя четверть часов
VALIDATION_PATH = os.path.join(ARTIFACTS, "validation.csv")
NOTE_CUTOUT = "ветер выше 25 м/с: принята гипотеза штатной остановки турбины (паспортный порог неизвестен), выработка 0"
CUTOUT_TEXT = ("принята гипотеза штатной остановки турбин (паспортный порог и параметры турбин неизвестны), "
               "выработка принята равной 0")


# ---------------------------------------------------------------- порог ширины коридора

@lru_cache(maxsize=1)
def _width_threshold_info() -> tuple[float, str]:
    """Порог ширины коридора и его источник ("validation" или "fallback"). Читается один раз на процесс:
    75-й процентиль p90 − p10 по январской проверке модели; если файла нет или он негоден, запасное 0.8."""
    reason = None
    if not os.path.exists(VALIDATION_PATH):
        reason = f"файла {VALIDATION_PATH} нет"
    else:
        try:
            val = pd.read_csv(VALIDATION_PATH)
            if not {"p10", "p90"} <= set(val.columns):
                reason = f"в {VALIDATION_PATH} нет колонок p10 и p90"
            else:
                width = (pd.to_numeric(val["p90"], errors="coerce")
                         - pd.to_numeric(val["p10"], errors="coerce")).dropna()
                if width.empty:
                    reason = f"в {VALIDATION_PATH} нет строк с p10 и p90"
                else:
                    value = round(float(width.quantile(WIDTH_QUANTILE)), 2)
                    if not np.isfinite(value):
                        reason = f"процентиль ширины по {VALIDATION_PATH} не число"
                    else:
                        log.info("Порог ширины коридора: %.2f, взят из %s (75-й процентиль p90 − p10 по %d строкам "
                                 "январской проверки модели)", value, VALIDATION_PATH, len(width))
                        return value, "validation"
        except Exception as exc:  # битый или пустой файл не должен ронять цикл
            reason = f"{VALIDATION_PATH} не читается ({type(exc).__name__}: {exc})"
    log.warning("Порог ширины коридора: запасное значение %.2f, %s", WIDTH_LOW_FALLBACK, reason)
    return WIDTH_LOW_FALLBACK, "fallback"


def width_threshold() -> float:
    """Порог ширины коридора p90 − p10 для низкой уверенности, округлён до 2 знаков."""
    return _width_threshold_info()[0]


def width_threshold_source() -> str:
    """Откуда взят порог: "validation" (январская проверка модели) или "fallback" (запасное значение)."""
    return _width_threshold_info()[1]


def threshold_text(analysis: dict) -> str:
    """Порог низкой уверенности словами, для журнала и шаблонной сводки."""
    if "width_threshold" in analysis:
        value, source = analysis["width_threshold"], analysis.get("width_threshold_source")
    else:  # анализ, сохранённый до появления порога в analysis: берём порог текущего процесса
        value, source = _width_threshold_info()
    if source == "validation":
        width_part = (f"ширина коридора выше {value:.2f}, это верхняя четверть часов по январской "
                      f"проверке модели")
    else:
        width_part = f"ширина коридора выше {value:.2f} (запасное значение, файла январской проверки нет)"
    return f"порог: {width_part}, либо сдвиг p50 к вчерашнему прогнозу больше {DELTA_P50_LOW}"


# ---------------------------------------------------------------- шаги цикла

def fetch_weather(issue_date: str) -> pd.DataFrame:
    """Прогноз погоды, известный в день issue_date, на следующие 48 часов (обёртка над model.weather)."""
    log.info("Погода: запрашиваю прогноз, известный %s, на %d часов", issue_date, HORIZON_HOURS)
    weather = get_issued_forecast(issue_date, HORIZON_HOURS)
    if len(weather) != HORIZON_HOURS:
        raise ValueError(
            f"Погода за {issue_date}: ожидалось {HORIZON_HOURS} часов, получено {len(weather)}"
        )
    source = _weather_source(weather)
    log.info(
        "Погода: %d часов с %s по %s, источник: %s, ветер на 100 м от %.1f до %.1f м/с",
        len(weather), weather.index[0], weather.index[-1], _source_ru(source),
        float(weather["ws100"].min()), float(weather["ws100"].max()),
    )
    return weather


def prepare(weather: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Признаки по каждой турбине из одного погодного ряда."""
    log.info("Признаки: строю по %d часам для турбин %s", len(weather), list(TURBINES))
    result = {}
    for turbine in TURBINES:
        features = build_features(weather, turbine)
        result[turbine] = features
        log.info("Признаки: турбина %d, %d строк, %d колонок", turbine, len(features), features.shape[1])
    return result


@lru_cache(maxsize=1)
def _models() -> dict | None:
    """Модели грузятся один раз на процесс: без артефактов load_models каждый раз заново строит
    кривую мощности по всей истории турбин. None, если predict не принимает models."""
    if "models" not in inspect.signature(predict).parameters:
        return None
    from model.predict import load_models
    return load_models()


def _predict(features: pd.DataFrame, turbine: int) -> pd.DataFrame:
    models = _models()
    if models is None:
        return predict(features, turbine)
    return predict(features, turbine, models=models)


def run_model(features_by_turbine: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Прогноз p10/p50/p90 обеих турбин в одной таблице. Индекс time; колонки turbine, p10, p50, p90,
    ws100, temp2m, note. В часы с ветром выше 25 м/с обнуляются все три квантили p10, p50, p90:
    гипотеза штатной остановки, паспортный порог турбин неизвестен."""
    parts = []
    for turbine, features in features_by_turbine.items():
        log.info("Модель: турбина %d, %d часов", turbine, len(features))
        pred = _predict(features, turbine).copy()
        pred["turbine"] = int(turbine)
        pred["ws100"] = features["ws100"].astype(float).values
        pred["temp2m"] = features["temp2m"].astype(float).values
        parts.append(pred)
    forecast = pd.concat(parts).sort_index(kind="stable")
    forecast = forecast.sort_values("turbine", kind="stable")
    forecast = forecast.sort_index(kind="stable")
    forecast.index.name = "time"  # дальше таблица сводится по (time, turbine)
    for col in ("p10", "p50", "p90"):
        forecast[col] = forecast[col].astype(float).clip(0.0, 1.0)
    quantiles = np.sort(forecast[["p10", "p50", "p90"]].to_numpy(), axis=1)
    forecast[["p10", "p50", "p90"]] = quantiles
    forecast["note"] = ""
    cutout = forecast["ws100"] > CUTOUT_WS100
    if cutout.any():
        forecast.loc[cutout, ["p10", "p50", "p90"]] = 0.0
        forecast.loc[cutout, "note"] = NOTE_CUTOUT
        log.info("Модель: %d часов с ветром выше %.0f м/с обнулены (гипотеза штатной остановки)",
                 int(cutout.sum()), CUTOUT_WS100)
    for turbine in TURBINES:
        sub = forecast[forecast["turbine"] == turbine]
        log.info("Модель: турбина %d, сумма p50 за 48 ч = %.2f, средний коридор %.2f",
                 turbine, float(sub["p50"].sum()), float((sub["p90"] - sub["p10"]).mean()))
    return forecast


def issue_timestamp(issue_date: str) -> pd.Timestamp:
    """Момент выпуска прогноза: день issue_date в 23:59 по местному времени (TZ)."""
    return pd.Timestamp(f"{issue_date} {ISSUE_TIME}", tz=TZ)


def weather_lead_frame(issue_date: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Упреждение погоды по часам погодного ряда. Индекс time; колонки weather_source,
    weather_lead_hours (24 для previous_day1, 48 для previous_day2, NaN, если упреждение неизвестно,
    например источник historical_forecast) и weather_run_time (target_time − упреждение, NaT, если неизвестно).
    Если колонки source в погоде нет, упреждение считается по дню часа относительно issue_date."""
    index = weather.index
    if "source" in weather.columns:
        sources = weather["source"].astype(str).tolist()
        leads = []
        for src in sources:
            k = src[len("previous_day"):] if src.startswith("previous_day") else ""
            leads.append(24.0 * int(k) if k.isdigit() else np.nan)
    else:
        issue_day = pd.Timestamp(issue_date).date()
        sources = ["по дню относительно выпуска"] * len(index)
        leads = [24.0 * (t.date() - issue_day).days for t in index]
    run_times = [t - pd.Timedelta(hours=lead) if np.isfinite(lead) else pd.NaT for t, lead in zip(index, leads)]
    run_time = pd.DatetimeIndex(pd.to_datetime(pd.Series(run_times, dtype=object), utc=True)).tz_convert(TZ)
    return pd.DataFrame({"weather_source": sources, "weather_lead_hours": leads,
                         "weather_run_time": run_time}, index=index)


def attach_weather_lead(issue_date: str, forecast: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Копия прогноза с колонками weather_source, weather_lead_hours, weather_run_time по индексу time."""
    lead = weather_lead_frame(issue_date, weather).reindex(forecast.index)
    out = forecast.copy()
    for col in ("weather_source", "weather_lead_hours", "weather_run_time"):
        out[col] = lead[col].values
    hours = lead[~lead.index.duplicated(keep="first")]["weather_lead_hours"]
    log.info("Погода: упреждение 24 ч у %d ч прогноза, 48 ч у %d ч, неизвестно у %d ч",
             int((hours == 24).sum()), int((hours == 48).sum()), int(hours.isna().sum()))
    return out


def save_forecast(issue_date: str, forecast: pd.DataFrame, weather: pd.DataFrame,
                  previous_forecast: pd.DataFrame | None = None) -> str:
    """CSV строго по колонкам контракта: 48 часов × 2 турбины = 96 строк. issue_timestamp — момент выпуска
    (D 23:59 местного), weather_lead_hours и weather_run_time — упреждение погоды и момент её расчёта.
    confidence = "low", если коридор шире порога width_threshold() или p50 ушёл от вчерашнего прогноза больше чем на 0.15."""
    os.makedirs(FORECASTS_DIR, exist_ok=True)
    path = os.path.join(FORECASTS_DIR, f"forecast_{issue_date}.csv")
    issue_start = pd.Timestamp(issue_date, tz=TZ)
    if "weather_lead_hours" not in forecast.columns or "weather_run_time" not in forecast.columns:
        forecast = attach_weather_lead(issue_date, forecast, weather)
    low = _low_confidence_mask(forecast, previous_forecast)
    w_lead = pd.to_numeric(forecast["weather_lead_hours"], errors="coerce")
    w_run = pd.to_datetime(forecast["weather_run_time"], errors="coerce", utc=True)
    rows = pd.DataFrame({
        "issue_date": issue_date,
        "target_time": [t.isoformat() for t in forecast.index],
        "lead_hours": [int(round((t - issue_start).total_seconds() / 3600)) for t in forecast.index],
        "issue_timestamp": issue_timestamp(issue_date).isoformat(),
        # пустое значение, если упреждение неизвестно (источник historical_forecast)
        "weather_lead_hours": [int(v) if np.isfinite(v) else "" for v in w_lead.to_numpy(dtype=float)],
        "weather_run_time": [t.tz_convert(TZ).isoformat() if pd.notna(t) else "" for t in w_run],
        "turbine": forecast["turbine"].astype(int).values,
        "ws100_forecast": forecast["ws100"].round(2).values,
        "temp_forecast": forecast["temp2m"].round(1).values,
        "p10": forecast["p10"].round(4).values,
        "p50": forecast["p50"].round(4).values,
        "p90": forecast["p90"].round(4).values,
        "confidence": np.where(low, "low", "ok"),
        "note": forecast["note"].values,
    })
    rows = rows[FORECAST_COLUMNS]
    if len(rows) != HORIZON_HOURS * len(TURBINES):
        raise ValueError(f"Прогноз {issue_date}: ожидалось {HORIZON_HOURS * len(TURBINES)} строк, получено {len(rows)}")
    rows.to_csv(path, index=False)
    log.info("Файл: %s, %d строк, часов низкой уверенности: %d", path, len(rows), int(low.sum()))
    return path


def analyze(issue_date: str, forecast: pd.DataFrame, previous_forecast: pd.DataFrame | None,
            actuals: pd.DataFrame | None) -> dict:
    """Правила анализа дня. Возвращает словарь только с числами и списками (для JSON, журнала и LLM)."""
    days = sorted({t.strftime("%Y-%m-%d") for t in forecast.index})
    totals = {}
    for turbine in TURBINES:
        sub = forecast[forecast["turbine"] == turbine]
        by_day = {d: round(float(sub[sub.index.strftime("%Y-%m-%d") == d]["p50"].sum()), 3) for d in days}
        totals[str(turbine)] = {
            "by_day": by_day,
            "total": round(float(sub["p50"].sum()), 3),
            "p10_total": round(float(sub["p10"].sum()), 3),
            "p90_total": round(float(sub["p90"].sum()), 3),
            "peak_hour": sub["p50"].idxmax().isoformat() if len(sub) else None,
            "peak_p50": round(float(sub["p50"].max()), 3) if len(sub) else None,
        }
    totals["all"] = {
        "by_day": {d: round(sum(totals[str(t)]["by_day"][d] for t in TURBINES), 3) for d in days},
        "total": round(sum(totals[str(t)]["total"] for t in TURBINES), 3),
    }

    delta = _delta_vs_previous(forecast, previous_forecast)

    low_mask = _low_confidence_mask(forecast, previous_forecast)
    width = forecast["p90"] - forecast["p10"]
    threshold = width_threshold()
    low_hours = []
    for (t, row), is_low, w in zip(forecast.iterrows(), low_mask, width):
        if not is_low:
            continue
        reasons = []
        if w > threshold:
            reasons.append(f"коридор {w:.2f} выше порога {threshold:.2f}")
        prev_p50 = _previous_p50(previous_forecast, t, int(row["turbine"]))
        if prev_p50 is not None and abs(float(row["p50"]) - prev_p50) > DELTA_P50_LOW:
            reasons.append(f"изменение к вчерашнему {float(row['p50']) - prev_p50:+.2f}")
        low_hours.append({
            "target_time": t.isoformat(), "turbine": int(row["turbine"]),
            "p50": round(float(row["p50"]), 3), "width": round(float(w), 3), "reason": ", ".join(reasons),
        })

    extreme = forecast[(forecast["turbine"] == min(TURBINES)) & (forecast["ws100"] > CUTOUT_WS100)]
    extreme_hours = [{"target_time": t.isoformat(), "ws100": round(float(r["ws100"]), 1)}
                     for t, r in extreme.iterrows()]

    error = _error_yesterday(issue_date, previous_forecast, actuals)

    first_turbine = forecast[forecast["turbine"] == min(TURBINES)]
    weather_summary = {
        "ws100_min": round(float(first_turbine["ws100"].min()), 1),
        "ws100_mean": round(float(first_turbine["ws100"].mean()), 1),
        "ws100_max": round(float(first_turbine["ws100"].max()), 1),
        "temp_min": round(float(first_turbine["temp2m"].min()), 1),
        "temp_max": round(float(first_turbine["temp2m"].max()), 1),
    }

    analysis = {
        "issue_date": issue_date,
        "days": days,
        "hours": int(len(first_turbine)),
        "turbines": [int(t) for t in TURBINES],
        "weather": weather_summary,
        "totals": totals,
        "delta_vs_previous": delta,
        "low_confidence_hours": low_hours,
        "low_confidence_count": len(low_hours),
        "width_threshold": threshold,
        "width_threshold_source": width_threshold_source(),
        "extreme_wind_hours": extreme_hours,
        "error_yesterday": error,
        "weather_lead_check": _weather_lead_check(issue_date, forecast),
    }
    log.info("Анализ: %s", analysis["weather_lead_check"]["text"])
    log.info(
        "Анализ: сумма p50 обеих турбин %.2f (по суткам %s), низкой уверенности %d ч, экстремального ветра %d ч, "
        "сравнение с вчера: %s, ошибка за вчера: %s",
        totals["all"]["total"], totals["all"]["by_day"], len(low_hours), len(extreme_hours),
        "нет вчерашнего прогноза" if delta is None else f"{delta['hours']} общих часов, средний сдвиг p50 {delta['mean_delta_p50']:+.3f}",
        "нет факта" if error is None else f"MAE {error['mae']:.3f}",
    )
    return analysis


def write_journal(issue_date: str, analysis: dict, note: str) -> None:
    """Добавляет раздел за день в output/journal.md; если раздел за эту дату уже есть, заменяет его."""
    os.makedirs(OUTPUT, exist_ok=True)
    section = _journal_section(issue_date, analysis, note)
    header = f"## Выпуск {issue_date}"
    existing = ""
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, encoding="utf-8") as f:
            existing = f.read()
    if not existing.strip():
        existing = "# Журнал агента\n\nКаждый раздел: один день выпуска прогноза на следующие 48 часов, обе турбины.\n"
    if header in existing:
        start = existing.index(header)
        nxt = existing.find("\n## Выпуск ", start + len(header))
        end = len(existing) if nxt == -1 else nxt + 1
        existing = existing[:start] + section + existing[end:]
        action = "заменён"
    else:
        if not existing.endswith("\n"):
            existing += "\n"
        existing += "\n" + section
        action = "добавлен"
    with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
        f.write(existing)
    log.info("Журнал: раздел за %s %s в %s", issue_date, action, JOURNAL_PATH)


# ---------------------------------------------------------------- вспомогательное

def load_previous_forecast(issue_date: str) -> pd.DataFrame | None:
    """Прогноз, выпущенный днём раньше, из output/forecasts, если файл есть. Формат как у run_model."""
    prev_date = (pd.Timestamp(issue_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    path = os.path.join(FORECASTS_DIR, f"forecast_{prev_date}.csv")
    if not os.path.exists(path):
        return None
    return read_forecast_csv(path)


def read_forecast_csv(path: str) -> pd.DataFrame:
    """Читает CSV прогноза обратно в формат run_model (индекс time, turbine, p10, p50, p90, ws100, temp2m, note);
    если в файле есть weather_lead_hours, weather_run_time, issue_timestamp, они тоже переносятся."""
    rows = pd.read_csv(path, keep_default_na=False)
    time = pd.DatetimeIndex(pd.to_datetime(rows["target_time"], utc=True), name="time").tz_convert(TZ)
    out = pd.DataFrame({
        "turbine": rows["turbine"].astype(int).values,
        "p10": rows["p10"].astype(float).values,
        "p50": rows["p50"].astype(float).values,
        "p90": rows["p90"].astype(float).values,
        "ws100": rows["ws100_forecast"].astype(float).values,
        "temp2m": rows["temp_forecast"].astype(float).values,
        "note": rows["note"].astype(str).values,
    }, index=time)
    # Новые колонки упреждения погоды: читаются, если есть; старые файлы без них тоже читаются.
    if "weather_lead_hours" in rows.columns:
        out["weather_lead_hours"] = pd.to_numeric(rows["weather_lead_hours"], errors="coerce").values
    if "weather_run_time" in rows.columns:
        run_time = pd.to_datetime(rows["weather_run_time"].replace("", None), errors="coerce", utc=True)
        out["weather_run_time"] = pd.DatetimeIndex(run_time).tz_convert(TZ)
    if "issue_timestamp" in rows.columns:
        out["issue_timestamp"] = rows["issue_timestamp"].astype(str).values
    return out


@lru_cache(maxsize=1)
def _hourly_actuals() -> pd.DataFrame:
    from model.prepare import load_hourly
    return load_hourly()


def load_actuals(issue_date: str) -> pd.DataFrame | None:
    """Факт выработки за день issue_date из истории турбин, если он есть (январь 2026); иначе None."""
    try:
        hourly = _hourly_actuals()
    except Exception as exc:  # файлов истории может не быть в чужой среде
        log.warning("Факт: история турбин недоступна (%s), ошибка за вчера не считается", exc)
        return None
    time = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="time")
    time = time.tz_localize(TZ) if time.tz is None else time.tz_convert(TZ)
    # Через DatetimeIndex, а не .values: .values теряет пояс и отдаёт UTC, что сдвигало факт на 5 часов.
    day = np.asarray(time.strftime("%Y-%m-%d") == issue_date) & hourly["power"].notna().to_numpy()
    if not day.any():
        return None
    sub = hourly.loc[day, ["turbine", "power"]].copy()
    sub.index = time[day]
    return sub


def _weather_source(weather: pd.DataFrame) -> str:
    if "fetched_from" in weather.columns and len(weather):
        return str(weather["fetched_from"].iloc[0])
    return "unknown"


def _source_ru(source: str) -> str:
    return {"network": "Open-Meteo, сеть", "cache": "архив прогнозов из кэша"}.get(source, source)


_MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
               "октября", "ноября", "декабря")


def _ru_day(value) -> str:
    """«2026-02-13» или «2026-02-13T20:00:00+05:00» → «13 февраля»."""
    text = str(value)
    try:
        return f"{int(text[8:10])} {_MONTHS_GEN[int(text[5:7]) - 1]}"
    except (ValueError, IndexError):
        return text


def _ru_time(value, sep: str = " ") -> str:
    """«2026-02-13T20:00:00+05:00» → «13 февраля 20:00» (sep=" в " даёт «13 февраля в 20:00»).
    Время местное, как записано в самой строке; смещение не показывается."""
    text = str(value)
    if len(text) >= 16 and text[10] in "T ":
        return f"{_ru_day(text)}{sep}{text[11:16]}"
    return _ru_day(text)


def _hours_word(n: int) -> str:
    """Склонение: 1 час, 2 часа, 5 часов, 21 час, 24 часа."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return "час"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "часа"
    return "часов"


def note_source_ru(note_source: str | None) -> str:
    """«model:gpt-5.5» → «модель gpt-5.5», «template» → «шаблон»."""
    if note_source and str(note_source).startswith("model:"):
        return "модель " + str(note_source)[len("model:"):]
    return "шаблон"


def _previous_p50(previous: pd.DataFrame | None, t: pd.Timestamp, turbine: int) -> float | None:
    if previous is None:
        return None
    sub = previous[(previous["turbine"] == turbine) & (previous.index == t)]
    if sub.empty:
        return None
    return float(sub["p50"].iloc[0])


def _low_confidence_mask(forecast: pd.DataFrame, previous: pd.DataFrame | None) -> np.ndarray:
    width_low = (forecast["p90"] - forecast["p10"]).to_numpy() > width_threshold()
    if previous is None:
        return width_low
    prev = previous.reset_index().set_index(["time", "turbine"])["p50"]
    key = pd.MultiIndex.from_arrays([forecast.index, forecast["turbine"].astype(int)])
    prev_aligned = prev.reindex(key).to_numpy()
    delta = np.abs(forecast["p50"].to_numpy() - prev_aligned)
    delta_low = np.where(np.isnan(delta), False, delta > DELTA_P50_LOW)
    return width_low | delta_low


def _delta_vs_previous(forecast: pd.DataFrame, previous: pd.DataFrame | None) -> dict | None:
    if previous is None:
        return None
    cur = forecast.reset_index().set_index(["time", "turbine"])[["p50"]]
    prev = previous.reset_index().set_index(["time", "turbine"])[["p50"]]
    common = cur.index.intersection(prev.index)
    if len(common) == 0:
        return {"hours": 0, "note": "общих часов с вчерашним прогнозом нет"}
    c = cur.loc[common, "p50"]
    p = prev.loc[common, "p50"]
    diff = c - p
    per_turbine = {}
    for turbine in TURBINES:
        idx = [k for k in common if k[1] == turbine]
        if not idx:
            continue
        per_turbine[str(turbine)] = {
            "sum_p50_new": round(float(c.loc[idx].sum()), 3),
            "sum_p50_prev": round(float(p.loc[idx].sum()), 3),
            "delta_energy": round(float(diff.loc[idx].sum()), 3),
            "mean_delta_p50": round(float(diff.loc[idx].mean()), 3),
            "max_abs_delta_p50": round(float(diff.loc[idx].abs().max()), 3),
        }
    hours = int(len({k[0] for k in common}))
    return {
        "hours": hours,
        "overlap_start": min(k[0] for k in common).isoformat(),
        "overlap_end": max(k[0] for k in common).isoformat(),
        "sum_p50_new": round(float(c.sum()), 3),
        "sum_p50_prev": round(float(p.sum()), 3),
        "delta_energy": round(float(diff.sum()), 3),
        "mean_delta_p50": round(float(diff.mean()), 3),
        "mean_abs_delta_p50": round(float(diff.abs().mean()), 3),
        "hours_changed_over_threshold": int((diff.abs() > DELTA_P50_LOW).sum()),
        "per_turbine": per_turbine,
    }


def _error_yesterday(issue_date: str, previous: pd.DataFrame | None, actuals: pd.DataFrame | None) -> dict | None:
    """MAE вчерашнего прогноза (выпуск D−1, часы дня D) против факта за D, если и то и другое есть."""
    if previous is None or actuals is None or actuals.empty:
        return None
    prev_day = previous[previous.index.strftime("%Y-%m-%d") == issue_date]
    if prev_day.empty:
        return None
    a = actuals.reset_index().set_index(["time", "turbine"])["power"]
    p = prev_day.reset_index().set_index(["time", "turbine"])
    common = p.index.intersection(a.index)
    if len(common) == 0:
        return None
    err = (p.loc[common, "p50"] - a.loc[common]).astype(float)
    inside = ((a.loc[common] >= p.loc[common, "p10"]) & (a.loc[common] <= p.loc[common, "p90"])).mean()
    per_turbine = {}
    for turbine in TURBINES:
        idx = [k for k in common if k[1] == turbine]
        if idx:
            per_turbine[str(turbine)] = {
                "mae": round(float(err.loc[idx].abs().mean()), 3),
                "bias": round(float(err.loc[idx].mean()), 3),
                "sum_actual": round(float(a.loc[idx].sum()), 3),
                "sum_p50": round(float(p.loc[idx, "p50"].sum()), 3),
            }
    return {
        "day": issue_date,
        "hours": int(len({k[0] for k in common})),
        "mae": round(float(err.abs().mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "bias": round(float(err.mean()), 3),
        "coverage_p10_p90": round(float(inside), 3),
        "per_turbine": per_turbine,
    }


def _ru_moment(ts: pd.Timestamp) -> str:
    """Момент словами: «5 февраля 23:59»."""
    return _ru_time(ts.tz_convert(TZ).isoformat())


def _weather_lead_check(issue_date: str, forecast: pd.DataFrame) -> dict:
    """Проверка, что каждое значение погоды предсказано не позже момента выпуска:
    weather_run_time <= issue_timestamp по всем часам; неизвестное упреждение — тоже не пройдено."""
    issue_ts = issue_timestamp(issue_date)
    first = forecast[forecast["turbine"] == min(TURBINES)]
    total = int(len(first))
    tz_note = f"(местное время, UTC{issue_ts.strftime('%z')[:3]}:{issue_ts.strftime('%z')[3:]})"
    if "weather_lead_hours" in first.columns and "weather_run_time" in first.columns:
        lead = pd.to_numeric(first["weather_lead_hours"], errors="coerce")
        run_time = pd.to_datetime(first["weather_run_time"], errors="coerce", utc=True).dt.tz_convert(TZ)
        unknown = (lead.isna() | run_time.isna()).to_numpy()
        known = run_time[~unknown]
    else:  # прогноз без колонок упреждения (например, старый файл): проверить нечем
        unknown = np.ones(total, dtype=bool)
        known = pd.Series([], dtype=f"datetime64[ns, {TZ}]")
    max_run = known.max() if len(known) else None
    late = int((known > issue_ts).sum())
    n_unknown = int(unknown.sum())
    ok = total > 0 and late == 0 and n_unknown == 0
    if ok:
        text = (f"Все значения погоды предсказаны не позже момента выпуска: самое позднее {_ru_moment(max_run)} "
                f"при выпуске {_ru_moment(issue_ts)} {tz_note}")
    else:
        parts = []
        if late:
            parts.append(f"{late} ч из {total} предсказаны позже момента выпуска {_ru_moment(issue_ts)} "
                         f"(самое позднее {_ru_moment(max_run)}) {tz_note}")
        if n_unknown:
            if "weather_source" in first.columns:
                srcs = sorted({str(s) for s in first.loc[unknown, "weather_source"]})
            else:
                srcs = ["в прогнозе нет колонок упреждения"]
            src_text = ", ".join("архив прогнозов historical_forecast, момент расчёта в нём не записан"
                                 if s == "historical_forecast" else s for s in srcs)
            parts.append(f"упреждение погоды неизвестно для {n_unknown} ч из {total} (источник: {src_text}), "
                         f"подтвердить, что эти значения предсказаны не позже выпуска {_ru_moment(issue_ts)}, нельзя")
        if total == 0:
            parts.append("часов прогноза нет")
        text = "Проверка момента расчёта погоды не пройдена: " + "; ".join(parts)
    return {
        "issue_timestamp": issue_ts.isoformat(),
        "max_weather_run_time": max_run.isoformat() if max_run is not None and pd.notna(max_run) else None,
        "ok": bool(ok),
        "text": text,
    }


def _journal_section(issue_date: str, analysis: dict, note: str) -> str:
    totals = analysis["totals"]
    days = analysis["days"]
    lines = [f"## Выпуск {issue_date}", ""]
    src = analysis.get("weather_source", "unknown")
    lines.append(f"Погода: {_source_ru(src)}; прогноз на {' и '.join(_ru_day(d) for d in days)} "
                 f"({analysis['hours']} {_hours_word(analysis['hours'])}). "
                 f"Ветер на 100 м от {analysis['weather']['ws100_min']:.1f} до {analysis['weather']['ws100_max']:.1f} м/с, "
                 f"в среднем {analysis['weather']['ws100_mean']:.1f} м/с.")
    check = analysis.get("weather_lead_check")
    if check:
        lines.append("")
        lines.append(check["text"] + ".")
    lines.append("")
    lines.append("Итоги (сумма p50; сумма нормализованной мощности по часам, доля номинала × час):")
    lines.append("")
    lines.append("| Турбина | " + " | ".join(_ru_day(d) for d in days)
                 + " | Всего 48 ч | Сумма p10 … сумма p90 (сумма квантилей по часам, не интервал суточной энергии) |")
    lines.append("|---|" + "---|" * (len(days) + 2))
    for t in analysis["turbines"]:
        tt = totals[str(t)]
        lines.append(f"| {t} | " + " | ".join(f"{tt['by_day'][d]:.2f}" for d in days)
                     + f" | {tt['total']:.2f} | {tt['p10_total']:.2f} … {tt['p90_total']:.2f} |")
    lines.append("| Обе | " + " | ".join(f"{totals['all']['by_day'][d]:.2f}" for d in days)
                 + f" | {totals['all']['total']:.2f} | |")
    lines.append("")
    delta = analysis["delta_vs_previous"]
    if delta is None:
        lines.append("Изменения к вчерашнему прогнозу: вчерашнего прогноза нет, сравнивать не с чем.")
    elif delta.get("hours", 0) == 0:
        lines.append("Изменения к вчерашнему прогнозу: общих часов нет.")
    else:
        lines.append(
            f"Изменения к вчерашнему прогнозу: {delta['hours']} {'общий' if _hours_word(delta['hours']) == 'час' else 'общих'} "
            f"{_hours_word(delta['hours'])} "
            f"({_ru_time(delta['overlap_start'])} … {_ru_time(delta['overlap_end'])}); сумма p50 обеих турбин была "
            f"{delta['sum_p50_prev']:.2f}, стала {delta['sum_p50_new']:.2f} ({delta['delta_energy']:+.2f}); "
            f"часов с изменением больше {DELTA_P50_LOW}: {delta['hours_changed_over_threshold']}."
        )
    lines.append("")
    low = analysis["low_confidence_hours"]
    if not low:
        lines.append(f"Часы низкой уверенности: нет ({threshold_text(analysis)}).")
    else:
        lines.append(f"Часы низкой уверенности ({len(low)}), {threshold_text(analysis)}:")
        lines.append("")
        for h in low[:24]:
            lines.append(f"- {_ru_time(h['target_time'])}, турбина {h['turbine']}: p50 {h['p50']:.2f}, {h['reason']}")
        if len(low) > 24:
            lines.append(f"- … и ещё {len(low) - 24}")
    lines.append("")
    ext = analysis["extreme_wind_hours"]
    if ext:
        lines.append(f"Экстремальный ветер выше {CUTOUT_WS100:.0f} м/с: {CUTOUT_TEXT}. Часы: "
                     + ", ".join(f"{_ru_time(h['target_time'])} ({h['ws100']} м/с)" for h in ext))
        lines.append("")
    err = analysis["error_yesterday"]
    if err is None:
        lines.append("Ошибка за вчера: факта за этот день в данных нет (или нет вчерашнего прогноза).")
    else:
        lines.append(f"Ошибка вчерашнего прогноза на {_ru_day(err['day'])} по факту: MAE {err['mae']:.3f}, RMSE {err['rmse']:.3f}, "
                     f"смещение {err['bias']:+.3f}, факт внутри коридора p10–p90 в {err['coverage_p10_p90'] * 100:.0f}% часов.")
    lines.append("")
    lines.append(f"Сводка дня (сводка: {note_source_ru(analysis.get('note_source'))}):")
    lines.append("")
    lines.append(note.strip())
    lines.append("")
    return "\n".join(lines) + "\n"
