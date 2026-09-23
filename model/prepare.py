# Подготовка данных турбин: чтение 10-минутных CSV, сведение к часу, фильтр неполных часов и простоев.

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import DATA_RAW, TZ

log = logging.getLogger("prepare")

# Колонки исходных CSV организаторов (по порядку): ID, время, ветер, мощность, температура.
RAW_COLUMNS = ["id", "time", "ws_measured", "power", "temp_measured"]


def _read_raw(turbine: int) -> pd.DataFrame:
    path = Path(DATA_RAW) / f"turbine_{turbine}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Нет файла данных турбины {turbine}: {path}")
    df = pd.read_csv(path, header=0, names=RAW_COLUMNS, usecols=range(5))
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    bad = df["time"].isna().sum()
    if bad:
        log.warning("Турбина %d: %d строк с нечитаемым временем пропущены", turbine, bad)
        df = df.dropna(subset=["time"])
    for c in ("ws_measured", "power", "temp_measured"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Округляем до часа ещё в наивном времени, потом локализуем в Asia/Almaty. При переводе часов
    # 01.03.2024 час 23:00 29.02 повторяется: в данных он один, берём первый вариант (до перевода).
    df["hour"] = df["time"].dt.floor("h")
    df["hour"] = df["hour"].dt.tz_localize(TZ, ambiguous=np.ones(len(df), dtype=bool), nonexistent="shift_forward")
    df["turbine"] = turbine
    return df


def load_hourly() -> pd.DataFrame:
    """Читает data/raw/turbine_{1,2}.csv, сводит к часу. Колонки: time, turbine, power (среднее 0..1),
    ws_measured, temp_measured, n_samples (сколько 10-минутных замеров вошло, 0..6)."""
    parts = []
    for turbine in (1, 2):
        raw = _read_raw(turbine)
        g = raw.groupby("hour")
        hourly = pd.DataFrame({
            "power": g["power"].mean().clip(0, 1),
            "ws_measured": g["ws_measured"].mean(),
            "temp_measured": g["temp_measured"].mean(),
            "n_samples": g["power"].count().clip(0, 6),
        })
        hourly.index.name = "time"
        hourly = hourly.reset_index()
        hourly["turbine"] = turbine
        log.info("Турбина %d: %d замеров → %d часов, %s … %s", turbine, len(raw), len(hourly),
                 hourly["time"].min(), hourly["time"].max())
        parts.append(hourly)
    out = pd.concat(parts, ignore_index=True)
    return out[["time", "turbine", "power", "ws_measured", "temp_measured", "n_samples"]]


# В этих данных нормированная мощность при ветре никогда не равна ровно 0 (минимум около 0,01,
# видимо, собственные нужды), поэтому «нулём» считается мощность не выше ZERO_POWER.
ZERO_POWER = 0.01


def _downtime_mask(df: pd.DataFrame, min_ws: float = 6.0, min_run: int = 3) -> pd.Series:
    """Вероятный простой: ветер >= min_ws, а мощность не выше ZERO_POWER, и так не меньше min_run
    часов подряд (по каждой турбине отдельно, по соседним часам)."""
    mask = pd.Series(False, index=df.index)
    for turbine, part in df.groupby("turbine"):
        part = part.sort_values("time")
        cand = (part["ws_measured"] >= min_ws) & (part["power"] <= ZERO_POWER)
        # Разрывы во времени рвут серию: серия считается только по подряд идущим часам.
        gap = part["time"].diff() != pd.Timedelta(hours=1)
        group_id = ((cand != cand.shift()) | gap).cumsum()
        run_len = cand.groupby(group_id).transform("size")
        mask.loc[part.index] = cand & (run_len >= min_run)
    return mask


def filter_training(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Убирает часы с n_samples < 4 и вероятные простои (ws_measured >= 6 м/с при power == 0
    не менее 3 часов подряд). Возвращает данные и отчёт: сколько часов убрано по каждой причине."""
    report = {"hours_total": int(len(df))}
    incomplete = df["n_samples"] < 4
    no_power = df["power"].isna()
    downtime = _downtime_mask(df) & ~incomplete & ~no_power
    report["removed_incomplete_hours"] = int(incomplete.sum())
    report["removed_no_power_value"] = int((no_power & ~incomplete).sum())
    report["removed_downtime_hours"] = int(downtime.sum())
    kept = df[~incomplete & ~no_power & ~downtime].copy()
    report["hours_kept"] = int(len(kept))
    for turbine, part in kept.groupby("turbine"):
        report[f"hours_kept_turbine_{turbine}"] = int(len(part))
    log.info("Фильтр: всего %d часов, убрано неполных %d, без мощности %d, простоев %d, осталось %d",
             report["hours_total"], report["removed_incomplete_hours"], report["removed_no_power_value"],
             report["removed_downtime_hours"], report["hours_kept"])
    return kept.reset_index(drop=True), report
