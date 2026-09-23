# Признаки для модели: из погодного прогноза (ветер, порывы, направление, температура, давление)
# и календаря (час суток, месяц) делает таблицу FEATURES для одной турбины, индекс time сохраняется.

import numpy as np
import pandas as pd

FEATURES = ["ws10", "ws100", "gust10", "dir_sin", "dir_cos", "temp2m", "pressure",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "turbine"]

# Колонки погоды, без которых признаки не построить (имена из model/weather.py).
REQUIRED_WEATHER = ["ws10", "ws100", "gust10", "dir100", "temp2m", "pressure"]


def build_features(weather: pd.DataFrame, turbine: int) -> pd.DataFrame:
    """Из погодного ряда (индекс time, WEATHER_COLUMNS) делает таблицу с колонками FEATURES,
    индекс time сохраняется. Лишние колонки входа (lead_hours, source и прочие) не используются."""
    if weather is None or len(weather) == 0:
        raise ValueError("Пустой погодный ряд: признаки строить не из чего")
    missing = [c for c in REQUIRED_WEATHER if c not in weather.columns]
    if missing:
        raise ValueError(f"В погодном ряду нет нужных колонок: {missing}. "
                         f"Ожидаются колонки {REQUIRED_WEATHER}")
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise ValueError("Индекс погодного ряда должен быть временем (DatetimeIndex с часовым поясом)")
    try:
        turbine = int(turbine)
    except (TypeError, ValueError):
        raise ValueError(f"Номер турбины должен быть целым числом, получено: {turbine!r}")

    out = pd.DataFrame(index=weather.index)
    for col in ("ws10", "ws100", "gust10", "temp2m", "pressure"):
        out[col] = pd.to_numeric(weather[col], errors="coerce").astype(float)

    # Направление ветра — угол по кругу: 359° и 1° почти одно и то же, поэтому раскладываем на синус и косинус.
    rad = np.deg2rad(pd.to_numeric(weather["dir100"], errors="coerce").astype(float).to_numpy())
    out["dir_sin"] = np.sin(rad)
    out["dir_cos"] = np.cos(rad)

    # Час суток (местный, по индексу) и месяц тоже циклические: 23 часа рядом с 0, декабрь рядом с январём.
    hour = weather.index.hour.to_numpy()
    month = weather.index.month.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["turbine"] = turbine
    out.index.name = "time"
    return out[FEATURES]
