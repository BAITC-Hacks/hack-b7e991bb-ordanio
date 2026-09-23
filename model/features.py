# Признаки для модели: из погодного прогноза (ветер, порывы, направление, температура, давление),
# давности прогноза и календаря (час суток, месяц) делает таблицу FEATURES для одной турбины.

import numpy as np
import pandas as pd

FEATURES = ["ws10", "ws100", "gust10", "dir_sin", "dir_cos", "temp2m", "pressure",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "turbine",
            "lead_day", "ws100_cube", "ws100_smooth"]

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

    # Давность прогноза в сутках: 0 — архив самых свежих прогнозов, 1 — прогноз на завтра, 2 — на послезавтра.
    if "lead_day" in weather.columns:
        lead = pd.to_numeric(weather["lead_day"], errors="coerce").fillna(0).astype(int)
    elif "lead_hours" in weather.columns:
        lead = (pd.to_numeric(weather["lead_hours"], errors="coerce").fillna(0) // 24).astype(int)
    else:
        lead = pd.Series(0, index=weather.index)
    out["lead_day"] = lead.to_numpy()

    # Мощность ветра растёт как куб скорости, поэтому куб ветра на 100 м даём модели отдельно.
    out["ws100_cube"] = out["ws100"] ** 3
    # Сглаженный ветер: среднее прогноза на соседние часы t−1, t, t+1 (на краях по доступным). Прогноз часто
    # верно ловит порыв, но ошибается на час, сглаживание это смягчает. Каждая давность сглаживается отдельно.
    smooth = pd.Series(np.nan, index=range(len(out)))
    ws = pd.Series(out["ws100"].to_numpy())
    for _, pos in pd.Series(range(len(out))).groupby(out["lead_day"].to_numpy()):
        order = pos.to_numpy()[np.argsort(out.index[pos.to_numpy()], kind="stable")]
        smooth.iloc[order] = ws.iloc[order].rolling(3, center=True, min_periods=1).mean().to_numpy()
    out["ws100_smooth"] = smooth.to_numpy()

    out.index.name = "time"
    return out[FEATURES]
