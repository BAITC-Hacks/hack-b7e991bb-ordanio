# Погода из Open-Meteo: архив прогнозов для обучения и «прогноз, известный в день D» для агента.
# Каждый ответ кэшируется в data/weather_cache/*.json; при WEATHER_OFFLINE=1 или ошибке сети берётся кэш.

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from common.config import HORIZON_HOURS, TZ, WEATHER_CACHE, WEATHER_POINT

log = logging.getLogger("weather")

HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Имена переменных Open-Meteo и наши короткие имена колонок (порядок совпадает).
OPEN_METEO_VARS = ["wind_speed_10m", "wind_speed_100m", "wind_gusts_10m",
                   "wind_direction_100m", "temperature_2m", "surface_pressure"]
WEATHER_COLUMNS = ["ws10", "ws100", "gust10", "dir100", "temp2m", "pressure"]


def _offline() -> bool:
    return os.environ.get("WEATHER_OFFLINE", "0").strip() in ("1", "true", "yes")


def _cache_path(name: str) -> Path:
    return Path(WEATHER_CACHE) / f"{name}.json"


def _fetch_json(name: str, url: str, params: dict) -> tuple[dict, str]:
    """Возвращает (json, откуда): "cache" или "network". Сначала кэш, потом сеть, ответ сохраняется."""
    path = _cache_path(name)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f), "cache"
    if _offline():
        raise RuntimeError(
            f"Режим без сети (WEATHER_OFFLINE=1), а в кэше нет файла {path}. "
            "Снимите WEATHER_OFFLINE или положите файл в кэш.")
    try:
        log.info("Запрос к Open-Meteo: %s (%s … %s)", name, params.get("start_date"), params.get("end_date"))
        r = httpx.get(url, params=params, timeout=60.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # сеть на площадке может лечь, объясняем по-русски
        raise RuntimeError(f"Не удалось получить погоду из Open-Meteo ({name}): {e}. "
                           "В кэше этого периода тоже нет.") from e
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo вернул ответ без данных ({name}): {str(data)[:200]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data, "network"


def _to_local_index(times: list[str], utc_offset_seconds: int) -> pd.DatetimeIndex:
    """Open-Meteo отдаёт строки времени с одним постоянным смещением на весь ответ (для Asia/Almaty
    это UTC+5 даже для 2023 года, когда часы стояли на UTC+6). Переводим в UTC по их смещению,
    а затем в настоящий Asia/Almaty, чтобы время было физически верным."""
    naive = pd.to_datetime(times)
    utc = (naive - pd.Timedelta(seconds=int(utc_offset_seconds))).tz_localize("UTC")
    return utc.tz_convert(TZ)


def _base_params() -> dict:
    return {"latitude": WEATHER_POINT[0], "longitude": WEATHER_POINT[1],
            "wind_speed_unit": "ms", "timezone": TZ}


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    chunks = []
    for year in range(s.year, e.year + 1):
        a = max(s, date(year, 1, 1))
        b = min(e, date(year, 12, 31))
        if a <= b:
            chunks.append((a.isoformat(), b.isoformat()))
    return chunks


def _find_cached_chunk(start: str, end: str) -> str | None:
    """Ищет в кэше файл архива, покрывающий отрезок start..end (например, годовой файл для среза)."""
    year = start[:4]
    for p in sorted(Path(WEATHER_CACHE).glob(f"historical_{year}-*.json")):
        try:
            _, a, b = p.stem.split("_")
        except ValueError:
            continue
        if a <= start and b >= end:
            return p.stem
    return None


def get_training_weather(start: str, end: str) -> pd.DataFrame:
    """Архив прогнозов (historical-forecast-api) по WEATHER_POINT, часовой ряд start..end.
    Индекс: time (tz Asia/Almaty). Колонки: WEATHER_COLUMNS. Запросы режутся по годам, кэшируются."""
    frames = []
    for a, b in _year_chunks(start, end):
        name = f"historical_{a}_{b}"
        if not _cache_path(name).exists():
            covering = _find_cached_chunk(a, b)
            if covering:
                name = covering
        params = _base_params() | {"start_date": a, "end_date": b, "hourly": ",".join(OPEN_METEO_VARS)}
        data, origin = _fetch_json(name, HISTORICAL_URL, params)
        log.info("Архив прогнозов %s … %s: %s", a, b, "из кэша" if origin == "cache" else "из сети")
        h = data["hourly"]
        df = pd.DataFrame({col: h[var] for col, var in zip(WEATHER_COLUMNS, OPEN_METEO_VARS)},
                          index=_to_local_index(h["time"], data["utc_offset_seconds"]))
        frames.append(df)
    out = pd.concat(frames).astype(float)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index.name = "time"
    lo = pd.Timestamp(start, tz=TZ)
    hi = pd.Timestamp(end, tz=TZ) + pd.Timedelta(hours=23)
    return out.loc[(out.index >= lo) & (out.index <= hi)]


def get_issued_forecast(issue_date: str, horizon_hours: int = HORIZON_HOURS) -> pd.DataFrame:
    """Прогноз, известный в день issue_date (previous-runs-api): для часов issue_date+1 берётся
    previous_day1, для issue_date+2 — previous_day2. Ровно horizon_hours строк начиная с
    issue_date+1 00:00. Индекс time, колонки WEATHER_COLUMNS + lead_hours (int, часы от issue_date 00:00)
    + source ("previous_day1"/"previous_day2") + fetched_from ("network"/"cache")."""
    try:
        issue = date.fromisoformat(str(issue_date)[:10])
    except ValueError:
        raise ValueError(f"Дата выпуска прогноза должна быть в виде ГГГГ-ММ-ДД, получено: {issue_date!r}")
    if horizon_hours < 1:
        raise ValueError("Горизонт прогноза должен быть не меньше одного часа")
    n_days = int(np.ceil(horizon_hours / 24))
    first_day = issue + timedelta(days=1)
    last_day = issue + timedelta(days=n_days)
    variables = [f"{v}_previous_day{k}" for k in range(1, n_days + 1) for v in OPEN_METEO_VARS]
    params = _base_params() | {"start_date": first_day.isoformat(), "end_date": last_day.isoformat(),
                               "hourly": ",".join(variables)}
    name = f"issued_{issue.isoformat()}" if horizon_hours == HORIZON_HOURS \
        else f"issued_{issue.isoformat()}_{horizon_hours}h"
    data, origin = _fetch_json(name, PREVIOUS_RUNS_URL, params)
    log.info("Прогноз, известный %s: %s", issue.isoformat(), "из кэша" if origin == "cache" else "из сети")

    h = data["hourly"]
    idx = _to_local_index(h["time"], data["utc_offset_seconds"])
    start_ts = pd.Timestamp(first_day.isoformat(), tz=TZ)
    target_index = pd.date_range(start_ts, periods=horizon_hours, freq="h", name="time")

    out = pd.DataFrame(index=target_index, columns=WEATHER_COLUMNS, dtype=float)
    out["lead_hours"] = 0
    out["source"] = ""
    for k in range(1, n_days + 1):
        day = issue + timedelta(days=k)
        day_start = pd.Timestamp(day.isoformat(), tz=TZ)
        day_mask = (target_index >= day_start) & (target_index < day_start + pd.Timedelta(days=1))
        if not day_mask.any():
            continue
        block = pd.DataFrame({col: h.get(f"{var}_previous_day{k}") for col, var in zip(WEATHER_COLUMNS, OPEN_METEO_VARS)},
                             index=idx).astype(float)
        block = block[~block.index.duplicated(keep="first")].reindex(target_index[day_mask])
        source = f"previous_day{k}"
        if block["ws100"].isna().all():
            # Предыдущий запуск для этого дня пустой: берём архив прогнозов и честно помечаем источник.
            log.warning("previous_day%d для %s пустой, беру архив прогнозов (historical) как замену",
                        k, day.isoformat())
            block = get_training_weather(day.isoformat(), day.isoformat()).reindex(target_index[day_mask])
            source = "historical_forecast"
        out.loc[day_mask, WEATHER_COLUMNS] = block[WEATHER_COLUMNS].values
        out.loc[day_mask, "source"] = source
    issue_start = pd.Timestamp(issue.isoformat(), tz=TZ)
    out["lead_hours"] = ((out.index - issue_start) / pd.Timedelta(hours=1)).astype(int)
    out["fetched_from"] = origin
    out[WEATHER_COLUMNS] = out[WEATHER_COLUMNS].astype(float)
    return out
