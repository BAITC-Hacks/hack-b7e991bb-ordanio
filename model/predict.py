# Предсказание выработки: загрузка обученных квантильных моделей или запасная кривая мощности по ветру.

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common.config import ARTIFACTS

log = logging.getLogger("predict")

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}

_MODELS_CACHE: dict | None = None   # модели читаются с диска один раз на процесс


class PowerCurveModel:
    """Кривая мощности по бинам скорости ветра (ширина 0,5 м/с) из истории, отдельно по турбинам.
    В каждом бине запоминаются квантили мощности 0.1, 0.5, 0.9. Это и точка отсчёта, и запасной
    предсказатель, если обученных моделей нет. Ветер берётся из колонки ws100 прогноза."""

    BIN_WIDTH = 0.5
    MIN_BIN_COUNT = 5

    def __init__(self, wind_column: str = "ws100"):
        self.wind_column = wind_column
        self.curves: dict[int, pd.DataFrame] = {}   # turbine -> DataFrame(index=bin_left, cols p10/p50/p90)

    def fit(self, wind: pd.Series, power: pd.Series, turbine: pd.Series) -> "PowerCurveModel":
        df = pd.DataFrame({"wind": wind.values, "power": power.values, "turbine": turbine.values}).dropna()
        df["bin"] = np.floor(df["wind"] / self.BIN_WIDTH) * self.BIN_WIDTH
        for t, part in df.groupby("turbine"):
            g = part.groupby("bin")["power"]
            curve = pd.DataFrame({"p10": g.quantile(0.1), "p50": g.quantile(0.5), "p90": g.quantile(0.9),
                                  "n": g.size()})
            curve = curve[curve["n"] >= self.MIN_BIN_COUNT]
            self.curves[int(t)] = curve.sort_index()
            log.info("Кривая мощности турбины %d: %d бинов, ветер %.1f … %.1f м/с",
                     t, len(curve), curve.index.min(), curve.index.max())
        return self

    def _lookup(self, wind: np.ndarray, turbine: int, column: str) -> np.ndarray:
        curve = self.curves.get(int(turbine))
        if curve is None or curve.empty:
            # Нет истории по этой турбине: берём любую доступную, чтобы не оставить прогноз пустым.
            if not self.curves:
                raise RuntimeError("Кривая мощности не обучена")
            curve = next(iter(self.curves.values()))
        bins = np.floor(np.nan_to_num(wind, nan=0.0) / self.BIN_WIDTH) * self.BIN_WIDTH
        # Линейная интерполяция по центрам бинов; за краями держим крайнее значение.
        centers = curve.index.values + self.BIN_WIDTH / 2
        return np.interp(bins + self.BIN_WIDTH / 2, centers, curve[column].values)

    def predict(self, features: pd.DataFrame, turbine: int) -> pd.DataFrame:
        wind = features[self.wind_column].to_numpy(dtype=float)
        out = pd.DataFrame(index=features.index)
        out["turbine"] = int(turbine)
        for col in ("p10", "p50", "p90"):
            out[col] = self._lookup(wind, turbine, col)
        return _finalize(out)


def _finalize(out: pd.DataFrame) -> pd.DataFrame:
    """Клипует в [0,1] и упорядочивает квантили: p10 <= p50 <= p90."""
    q = np.sort(np.clip(out[["p10", "p50", "p90"]].to_numpy(dtype=float), 0.0, 1.0), axis=1)
    out[["p10", "p50", "p90"]] = q
    return out[["turbine", "p10", "p50", "p90"]]


def load_models(artifacts_dir: str | None = None, refresh: bool = False) -> dict:
    """{"q10": ..., "q50": ..., "q90": ...} из artifacts_dir (по умолчанию model/artifacts); если
    артефактов нет — PowerCurveModel (сохранённая кривая power_curve.joblib, а если и её нет — кривая,
    обученная тут же по истории). Папка по умолчанию кэшируется на процесс; refresh=True перечитывает
    с диска (после переобучения). Другая папка читается каждый раз."""
    global _MODELS_CACHE
    if artifacts_dir is not None:
        return _load_models_from_disk(Path(artifacts_dir))
    if _MODELS_CACHE is not None and not refresh:
        return _MODELS_CACHE
    _MODELS_CACHE = _load_models_from_disk(Path(ARTIFACTS))
    return _MODELS_CACHE


def _load_models_from_disk(art: Path) -> dict:
    paths = {k: art / f"model_{k}.joblib" for k in QUANTILES}
    if all(p.exists() for p in paths.values()):
        models = {k: joblib.load(p) for k, p in paths.items()}
        log.info("Загружены квантильные модели из %s", art)
        return models
    log.warning("Обученных моделей в %s нет, работает запасная кривая мощности по ветру", art)
    curve_path = art / "power_curve.joblib"
    if curve_path.exists():
        curve = joblib.load(curve_path)
    else:
        from model.prepare import filter_training, load_hourly
        from model.weather import get_training_weather
        from common.config import TRAIN_START
        hourly, _ = filter_training(load_hourly())
        weather = get_training_weather(TRAIN_START, hourly["time"].max().strftime("%Y-%m-%d"))
        joined = hourly.join(weather[["ws100"]], on="time", how="inner")
        curve = PowerCurveModel().fit(joined["ws100"], joined["power"], joined["turbine"])
    return {"q10": curve, "q50": curve, "q90": curve, "fallback": True}


def predict(features: pd.DataFrame, turbine: int, models: dict | None = None) -> pd.DataFrame:
    """Индекс time; колонки: turbine, p10, p50, p90 в [0,1], p10 <= p50 <= p90 (после сортировки)."""
    if models is None:
        models = load_models()
    if features is None or len(features) == 0:
        raise ValueError("Пустая таблица признаков: предсказывать нечего")
    if models.get("fallback"):
        return models["q50"].predict(features, turbine)
    from model.features import FEATURES
    missing = [c for c in FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"В признаках нет колонок: {missing}")
    X = features[FEATURES].to_numpy(dtype=float)
    out = pd.DataFrame(index=features.index)
    out["turbine"] = int(turbine)
    out["p10"] = models["q10"].predict(X)
    out["p50"] = models["q50"].predict(X)
    out["p90"] = models["q90"].predict(X)
    return _finalize(out)
