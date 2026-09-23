# Обучение прогноза выработки: факт турбин + архив прогнозов погоды → три квантильные модели (p10, p50, p90),
# самопроверка времени, проверка на январе 2026 так, как работает агент, артефакты и отчёт report.md.

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common.config import ARTIFACTS, TRAIN_START, TURBINES, TZ, VALIDATION_MONTH
from model.features import FEATURES, build_features
from model.predict import QUANTILES, PowerCurveModel, predict
from model.prepare import filter_training, load_hourly
from model.weather import get_issued_forecast, get_training_weather

log = logging.getLogger("train")

# День перевода часов в Казахстане на UTC+5. До него и после самопроверка времени идёт отдельно.
CLOCK_CHANGE = "2024-03-01"
SHIFTS = list(range(-2, 3))
MIN_ROWS_FOR_CORR = 100   # меньше строк в периоде: корреляцию не считаем, сдвиг берём 0

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "random_state": 0,
}

PERIOD_NAMES = {
    "before": f"до {CLOCK_CHANGE}",
    "after": f"с {CLOCK_CHANGE}",
    "all": "весь период",
}


# ---------------------------------------------------------------- время и выравнивание

def _validation_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Первый и последний день месяца проверки (даты без часового пояса)."""
    p = pd.Period(VALIDATION_MONTH, freq="M")
    return p.start_time.normalize(), p.end_time.normalize()


def _period_masks(times: pd.Series) -> dict[str, pd.Series]:
    cut = pd.Timestamp(CLOCK_CHANGE, tz=TZ)
    return {"before": times < cut, "after": times >= cut, "all": pd.Series(True, index=times.index)}


def _shift_series(s: pd.Series | pd.DataFrame, hours: int):
    """Сдвиг погоды на k часов: значение прогноза на час t+k встаёт в строку часа t."""
    out = s.copy()
    out.index = s.index - pd.Timedelta(hours=int(hours))
    return out


def check_alignment(facts: pd.DataFrame, weather: pd.DataFrame) -> dict:
    """Корреляция замеренного ветра (обе турбины вместе) с прогнозом ws100 при сдвигах −2..+2 часа,
    отдельно до перевода часов, после и в целом. Сдвиг k: замер в час t против прогноза на час t+k."""
    result = {}
    for key, mask in _period_masks(facts["time"]).items():
        part = facts.loc[mask, ["time", "ws_measured"]].dropna()
        by_shift = {}
        for k in SHIFTS:
            ws = _shift_series(weather["ws100"], k).rename("ws100_forecast")
            j = part.join(ws, on="time", how="inner")
            corr = float(j["ws_measured"].corr(j["ws100_forecast"])) if len(j) >= MIN_ROWS_FOR_CORR else None
            by_shift[str(k)] = {"corr": corr, "rows": int(len(j))}
        valid = {int(k): v["corr"] for k, v in by_shift.items() if v["corr"] is not None and not np.isnan(v["corr"])}
        best = max(valid, key=valid.get) if valid else None
        info = {"name": PERIOD_NAMES[key], "by_shift": by_shift, "best_shift": best,
                "second_shift": None, "margin_to_second": None, "peak_hours_estimate": None}
        if best is not None and len(valid) > 1:
            second = max((k for k in valid if k != best), key=valid.get)
            info["second_shift"] = second
            info["margin_to_second"] = valid[best] - valid[second]
        # Уточнение внутри часа: парабола через лучший сдвиг и двух соседей, её вершина.
        if best is not None and (best - 1) in valid and (best + 1) in valid:
            y_m, y_0, y_p = valid[best - 1], valid[best], valid[best + 1]
            denom = y_m - 2 * y_0 + y_p
            if denom != 0:
                info["peak_hours_estimate"] = best + 0.5 * (y_m - y_p) / denom
        result[key] = info
        if best is None:
            log.info("Самопроверка времени, %s: строк слишком мало, корреляцию не считаю", PERIOD_NAMES[key])
        else:
            log.info("Самопроверка времени, %s: корреляции по сдвигам %s; лучший сдвиг %+d "
                     "(замер в час t ближе всего к прогнозу на час t%+d)",
                     PERIOD_NAMES[key],
                     ", ".join(f"{k:+d}: {valid[k]:.4f}" for k in sorted(valid)), best, best)
    applied = {key: int(result[key]["best_shift"] or 0) for key in ("before", "after")}
    for key, k in applied.items():
        if k != 0:
            log.info("Сдвиг %+d ч применяю к погоде периода «%s» при склейке с фактом", k, PERIOD_NAMES[key])
    if not any(applied.values()):
        log.info("Лучший сдвиг в обоих периодах 0: погоду с фактом склеиваю час в час")
    return {"periods": result, "applied_shift_hours": applied}


def _shift_for_times(times: pd.Series, applied: dict) -> np.ndarray:
    cut = pd.Timestamp(CLOCK_CHANGE, tz=TZ)
    return np.where(times >= cut, applied["after"], applied["before"]).astype(int)


def join_features(facts: pd.DataFrame, weather: pd.DataFrame, applied: dict) -> pd.DataFrame:
    """Склейка факта с признаками из прогноза погоды по времени и турбине, с учётом сдвига периода."""
    parts = []
    masks = _period_masks(facts["time"])
    for key in ("before", "after"):
        f_part = facts[masks[key]]
        if f_part.empty:
            continue
        w = _shift_series(weather, applied[key])
        for turbine in sorted(f_part["turbine"].unique()):
            feats = build_features(w, int(turbine)).reset_index()
            one = f_part[f_part["turbine"] == turbine]
            parts.append(one.merge(feats, on=["time", "turbine"], how="inner"))
    if not parts:
        raise ValueError("После склейки факта с погодой не осталось ни одной строки")
    return pd.concat(parts, ignore_index=True).sort_values(["turbine", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------- метрики

def _mae(e: pd.Series) -> float | None:
    return float(np.mean(np.abs(e))) if len(e) else None


def _rmse(e: pd.Series) -> float | None:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else None


def _band(actual: pd.Series, lo: pd.Series, hi: pd.Series) -> tuple[float | None, float | None]:
    if not len(actual):
        return None, None
    inside = (actual >= lo) & (actual <= hi)
    return float(inside.mean()), float((hi - lo).mean())


def group_metrics(df: pd.DataFrame) -> dict:
    """Метрики одного среза: модель (p50, коридор p10–p90), persistence, кривая мощности."""
    cov, width = _band(df["actual"], df["p10"], df["p90"])
    c_cov, c_width = _band(df["actual"], df["curve_p10"], df["curve_p90"])
    has_pers = df["persistence"].notna()
    pers = df[has_pers]
    return {
        "rows": int(len(df)),
        "model": {"mae": _mae(df["p50"] - df["actual"]), "rmse": _rmse(df["p50"] - df["actual"]),
                  "coverage_p10_p90": cov, "mean_width_p10_p90": width,
                  "mae_on_persistence_rows": _mae(pers["p50"] - pers["actual"]),
                  "rmse_on_persistence_rows": _rmse(pers["p50"] - pers["actual"])},
        "persistence": {"rows": int(len(pers)), "rows_without_fact_for_day_d": int((~has_pers).sum()),
                        "mae": _mae(pers["persistence"] - pers["actual"]),
                        "rmse": _rmse(pers["persistence"] - pers["actual"])},
        "power_curve": {"mae": _mae(df["curve"] - df["actual"]), "rmse": _rmse(df["curve"] - df["actual"]),
                        "coverage_p10_p90": c_cov, "mean_width_p10_p90": c_width},
    }


# ---------------------------------------------------------------- проверка «как агент»

def validate(models: dict, curve: PowerCurveModel, facts: pd.DataFrame, applied: dict) -> tuple[pd.DataFrame, dict]:
    """Для каждой даты выпуска месяца проверки берёт прогноз погоды, известный в тот день, строит прогноз
    по обеим турбинам и сравнивает с фактом. Возвращает строки проверки и сводку по ним."""
    first_day, last_day = _validation_bounds()
    issue_dates = pd.date_range(first_day - pd.Timedelta(days=1), last_day - pd.Timedelta(days=1), freq="D")
    fact = facts.set_index(["time", "turbine"])["power"]
    frames = []
    for issue in issue_dates:
        issue_str = issue.strftime("%Y-%m-%d")
        fc = get_issued_forecast(issue_str)
        for turbine in sorted(TURBINES):
            feats = build_features(fc, turbine)
            p = predict(feats, turbine, models)
            c = curve.predict(feats, turbine)
            df = pd.DataFrame({
                "time": feats.index, "turbine": turbine, "issue_date": issue_str,
                "lead_hours": fc["lead_hours"].to_numpy(), "source": fc["source"].to_numpy(),
                "p10": p["p10"].to_numpy(), "p50": p["p50"].to_numpy(), "p90": p["p90"].to_numpy(),
                "curve": c["p50"].to_numpy(), "curve_p10": c["p10"].to_numpy(), "curve_p90": c["p90"].to_numpy(),
            })
            frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    rows["lead_day"] = rows["lead_hours"] // 24
    # Время факта с учётом найденного сдвига: прогноз на час t+k сравнивается с замером часа t.
    fact_time = rows["time"] - pd.to_timedelta(_shift_for_times(rows["time"], applied), unit="h")
    key = pd.MultiIndex.from_arrays([fact_time, rows["turbine"]])
    rows["actual"] = fact.reindex(key).to_numpy()
    # Persistence: для даты выпуска D берём факт дня D в тот же час.
    pers_time = fact_time - pd.to_timedelta(rows["lead_day"], unit="D")
    rows["persistence"] = fact.reindex(pd.MultiIndex.from_arrays([pers_time, rows["turbine"]])).to_numpy()

    summary = {
        "issue_dates": [issue_dates[0].strftime("%Y-%m-%d"), issue_dates[-1].strftime("%Y-%m-%d")],
        "n_issue_dates": int(len(issue_dates)),
        "forecast_rows": int(len(rows)),
        "sources": {str(k): int(v) for k, v in rows["source"].value_counts().items()},
    }
    no_fact = rows["actual"].isna()
    in_month = rows["time"].dt.strftime("%Y-%m") == VALIDATION_MONTH
    summary["rows_without_fact"] = int(no_fact.sum())
    summary["rows_without_fact_outside_month"] = int((no_fact & ~in_month).sum())
    summary["rows_without_fact_inside_month"] = int((no_fact & in_month).sum())
    rows = rows[~no_fact].reset_index(drop=True)
    summary["rows_with_fact"] = int(len(rows))
    per_hour = rows.groupby(["time", "turbine"]).size()
    summary["hours_seen_once"] = int((per_hour == 1).sum())
    summary["hours_seen_twice"] = int((per_hour == 2).sum())
    log.info("Проверка: %d дат выпуска, %d строк прогноза, с фактом %d, без факта %d (из них вне месяца %d)",
             summary["n_issue_dates"], summary["forecast_rows"], summary["rows_with_fact"],
             summary["rows_without_fact"], summary["rows_without_fact_outside_month"])
    return rows, summary


# ---------------------------------------------------------------- отчёт

def _f(x, digits: int = 4) -> str:
    return "нет данных" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _pct(x) -> str:
    return "нет данных" if x is None else f"{100 * x:.1f}%"


def _ratio_sentence(model_mae, other_mae, other_name: str) -> str:
    if model_mae is None or other_mae is None or other_mae == 0:
        return ""
    return (f"MAE модели составляет {100 * model_mae / other_mae:.1f}% от MAE {other_name} "
            f"на тех же строках.")


def write_report(path: Path, m: dict) -> None:
    d, al, v, mp = m["data"], m["time_alignment"], m["validation"], m["model"]
    per = al["periods"]
    L = []
    L.append("# Прогноз выработки ветростанции: как обучена модель и как она проверена\n")
    L.append("## Метод\n")
    L.append("Задача: для каждой из двух турбин дать прогноз средней за час выработки на 48 часов вперёд, "
             "начиная с полуночи следующего дня. Выработка везде нормирована: 0 означает ноль, 1 означает "
             "номинальную мощность турбины. На каждый час выдаются три числа. p50 это основная оценка "
             "(медиана). p10 и p90 это нижняя и верхняя граница коридора, в который факт по замыслу должен "
             "попадать примерно в 80% часов.\n")
    L.append("Входы модели берутся только из прогноза погоды Open-Meteo в одной точке между турбинами "
             "(турбины стоят в 240 м друг от друга и попадают в одну ячейку сетки погодной модели) и из "
             "календаря. Модель учится на парах «прогноз погоды на этот час» и «фактическая выработка в этот "
             "час» за всю историю турбин, кроме месяца проверки.\n")
    L.append("Сама модель это градиентный бустинг решающих деревьев (HistGradientBoostingRegressor из "
             "scikit-learn). Она состоит из нескольких сотен небольших деревьев вида «если прогноз ветра на "
             "100 м больше X и ветер дует с такого-то направления, то добавить к оценке столько-то». Каждое "
             "следующее дерево поправляет ошибку уже построенных. Обучены три отдельные модели, по одной на "
             "p10, p50 и p90. Для p10 ошибка считается так, что модели выгодно оказаться выше факта примерно "
             "в 10% часов; для p90 примерно в 90%; для p50 в половине часов.\n")
    L.append("Кривая мощности по прогнозу ветра на 100 м учитывает только скорость ветра. Прогноз погоды "
             "ошибается неодинаково при разных направлениях ветра, в разные сезоны и в разное время суток, "
             "а выработка зависит ещё от порывов и плотности воздуха (температура и давление). Бустинг "
             "находит такие поправки сам по истории. Кривая мощности оставлена рядом как точка отсчёта и "
             "как запасной предсказатель, если обученных моделей нет.\n")
    L.append("Проверка устроена так же, как работает агент. Месяц проверки в обучение не входил. Для каждой "
             f"даты выпуска с {v['issue_dates'][0]} по {v['issue_dates'][1]} взят прогноз погоды, известный в "
             "день выпуска (сервис previous-runs Open-Meteo: на завтра прогноз предыдущих суток, на "
             "послезавтра прогноз двухсуточной давности). По нему построен прогноз выработки и сравнён с "
             "фактом.\n")
    L.append("Рядом посчитаны две точки отсчёта. Persistence: «завтра и послезавтра будет как сегодня», "
             "для даты выпуска D берётся факт дня D в тот же час. Кривая мощности: выработка по прогнозу "
             "ветра на 100 м через кривую, построенную по тем же обучающим часам.\n")

    L.append("## Данные\n")
    L.append("| Показатель | Турбина 1 | Турбина 2 | Всего |")
    L.append("|---|---|---|---|")
    L.append(f"| Часов после сведения 10-минутных замеров к часу | {d['hours_total_turbine_1']} | "
             f"{d['hours_total_turbine_2']} | {d['hours_total']} |")
    L.append(f"| Осталось после фильтра | {d.get('hours_kept_turbine_1', 0)} | {d.get('hours_kept_turbine_2', 0)} | "
             f"{d['hours_kept']} |")
    L.append(f"| Обучающих строк (с погодой, без месяца проверки) | {d['train_rows_turbine_1']} | "
             f"{d['train_rows_turbine_2']} | {d['train_rows']} |")
    L.append("")
    L.append(f"Факт взят с {d['facts_first']} по {d['facts_last']}. Фильтр убрал {d['removed_incomplete_hours']} "
             f"неполных часов (меньше 4 из 6 десятиминутных замеров), {d['removed_no_power_value']} часов без "
             f"значения мощности и {d['removed_downtime_hours']} часов вероятного простоя (замеренный ветер "
             "6 м/с и больше при мощности не выше 0,01 номинала три часа подряд и дольше). "
             f"Часов факта, для которых в архиве прогнозов погоды нет строки: {d['rows_without_weather']}; "
             f"они в обучение не вошли. Часов факта в месяце проверки {VALIDATION_MONTH} по двум турбинам: "
             f"{d['validation_month_rows']}; они тоже отложены и в обучение не вошли.\n")
    L.append(f"Архив прогнозов погоды: часов {m['weather']['hours']}, с {m['weather']['first']} по "
             f"{m['weather']['last']}, пустых значений {m['weather']['nan_values']}.\n")

    L.append("## Время и самопроверка сдвига\n")
    L.append("Open-Meteo отдаёт время всех дат с одним постоянным смещением UTC+5, в том числе для 2023 года, "
             "когда в Казахстане действовало UTC+6. Модуль погоды переводит эти строки сначала в UTC, потом в "
             f"настоящее местное время Asia/Almaty. {CLOCK_CHANGE} Казахстан перевёл часы на UTC+5. В данных "
             "турбин повтора часа в эту ночь нет, поэтому часы SCADA могли остаться на прежнем времени. "
             "Поэтому самопроверка сделана отдельно до перевода и после.\n")
    L.append("Самопроверка сравнивает замеренный на турбинах ветер (обе турбины вместе, после фильтра) с "
             "прогнозом ветра на 100 м при сдвигах от −2 до +2 часов. Сдвиг k означает: замер турбины в час t "
             "сравнивается с прогнозом на час t+k. В таблице корреляция и число пар.\n")
    L.append("| Сдвиг k, ч | " + " | ".join(per[p]["name"] for p in ("before", "after", "all")) + " |")
    L.append("|---|---|---|---|")
    for k in SHIFTS:
        cells = []
        for p in ("before", "after", "all"):
            c = per[p]["by_shift"][str(k)]
            cells.append(f"{_f(c['corr'])} ({c['rows']})")
        L.append(f"| {k:+d} | " + " | ".join(cells) + " |")
    L.append("")
    for p in ("before", "after", "all"):
        info = per[p]
        if info["best_shift"] is None:
            L.append(f"- {info['name']}: данных мало, сдвиг не определялся.")
            continue
        b = info["best_shift"]
        s = (f"- {info['name']}: лучший сдвиг {b:+d} ч. Замер турбины в час t лучше всего совпадает с "
             f"прогнозом на час t{b:+d}.")
        if info["second_shift"] is not None:
            s += (f" Следующий по качеству сдвиг {info['second_shift']:+d} ч, разница корреляций "
                  f"{_f(info['margin_to_second'])}.")
        if info["peak_hours_estimate"] is not None:
            s += (f" Если провести параболу через три соседние точки, её вершина приходится на "
                  f"{info['peak_hours_estimate']:+.2f} ч.")
        L.append(s)
    L.append("")
    pb, pa = per["before"]["peak_hours_estimate"], per["after"]["peak_hours_estimate"]
    if pb is not None and pa is not None:
        diff = pb - pa
        text = (f"Вершины двух периодов отличаются на {diff:+.2f} ч. Замер SCADA это среднее за час. Прогноз "
                "погоды Open-Meteo дан на момент начала часа. Поэтому вершина около ±0,5 ч ожидаема сама по себе.")
        if 0.5 <= abs(diff) <= 1.5:
            text += (" Разница между периодами близка к одному часу. Это согласуется с тем, что часы SCADA после "
                     f"{CLOCK_CHANGE} не переводились и остались на UTC+6. Целый сдвиг в каждом периоде выбран "
                     "по наибольшей корреляции. Дробную часть часа сдвиг на целые часы исправить не может.\n")
        else:
            text += (" Разница между периодами меньше половины часа или больше полутора часов. Признаков "
                     "непереведённых часов SCADA самопроверка не показала.\n")
        L.append(text)
    ap = al["applied_shift_hours"]
    if any(ap.values()):
        L.append(f"Применённые сдвиги: {PERIOD_NAMES['before']} {ap['before']:+d} ч, {PERIOD_NAMES['after']} "
                 f"{ap['after']:+d} ч. При обучении прогноз погоды на час t+k ставится в строку замера часа t. "
                 "При проверке на январе прогноз на час t сравнивается с фактом часа t−k.\n")
        if ap["after"] != 0:
            L.append("Важно: агент и model/predict.py этот сдвиг сами не применяют. Прогноз агента на час t "
                     f"соответствует часу t{ap['after']:+d} по часам SCADA.\n")
    else:
        L.append("Лучший целый сдвиг в обоих периодах равен 0, поэтому погода и факт склеены час в час, "
                 "без сдвига.\n")

    L.append("## Признаки\n")
    L.append("Модель получает 12 признаков: скорость ветра на 10 и 100 м, порывы на 10 м, направление ветра "
             "на 100 м (синус и косинус угла, чтобы 359° и 1° были рядом), температура воздуха на 2 м, "
             "давление у поверхности, местный час суток и месяц (тоже синус и косинус, чтобы 23 часа были "
             "рядом с полуночью, а декабрь рядом с январём), номер турбины.\n")
    L.append("Все погодные признаки взяты из прогноза. Замеры турбины (ветер на гондоле, температура) в "
             "признаки не входят: в день выпуска прогноза замеров за завтра и послезавтра ещё нет. Если учить "
             "модель на замерах, она привыкнет к точному ветру и будет ошибаться сильнее на прогнозном.\n")

    L.append("## Параметры моделей\n")
    params = mp["params"]
    L.append(f"HistGradientBoostingRegressor, loss=\"quantile\", квантили {', '.join(str(q) for q in mp['quantiles'].values())}. "
             f"Параметры одинаковые для всех трёх: число деревьев {params['max_iter']}, шаг обучения "
             f"{params['learning_rate']}, листьев в дереве не больше {params['max_leaf_nodes']}, в листе не меньше "
             f"{params['min_samples_leaf']} часов, регуляризация L2 {params['l2_regularization']}, ранняя "
             f"остановка {'включена' if params['early_stopping'] else 'выключена'}, random_state="
             f"{params['random_state']}. Параметры заданы заранее и не подбирались по январю. Обучение трёх "
             f"моделей и кривой мощности заняло {mp['train_seconds']:.1f} с на процессоре.\n")
    L.append(f"Кривая мощности: бины прогноза ветра на 100 м шириной {PowerCurveModel.BIN_WIDTH} м/с, в каждом "
             "бине квантили выработки 0.1, 0.5 и 0.9, отдельно по турбинам.\n")

    L.append(f"## Проверка на {VALIDATION_MONTH}\n")
    L.append(f"Дат выпуска {v['n_issue_dates']}, строк прогноза {v['forecast_rows']} (48 часов × 2 турбины на каждую "
             f"дату). Строк, сопоставленных с фактом: {v['rows_with_fact']}. Строк без факта: "
             f"{v['rows_without_fact']}. Из них за пределами месяца проверки (день 2 последней даты выпуска): "
             f"{v['rows_without_fact_outside_month']}; внутри месяца (час убран фильтром или замеров нет): "
             f"{v['rows_without_fact_inside_month']}. Часов турбины, которые встречаются в проверке дважды "
             f"(как день 1 и как день 2): {v['hours_seen_twice']}; один раз: {v['hours_seen_once']}. Первый день "
             "месяца покрыт только как день 1, потому что прогноза с датой выпуска накануне в кэше нет. "
             "Источник погоды по строкам: "
             + ", ".join(f"{k} {n}" for k, n in v["sources"].items()) + ".\n")
    L.append("Все ошибки в долях номинальной мощности: 0.1 означает 10% номинала. Покрытие это доля часов, "
             "где факт попал в коридор от p10 до p90 включительно; по замыслу около 0.8. Ширина это средняя "
             "разница p90 − p10. Persistence не считается там, где нет факта за день выпуска в тот же час; "
             "такие строки указаны отдельно, а для честного сравнения рядом дана ошибка модели на тех же "
             "строках.\n")
    L.append("| Срез | Метод | Строк | MAE | RMSE | Покрытие p10–p90 | Ширина коридора |")
    L.append("|---|---|---|---|---|---|---|")
    groups = [("Все", v["overall"])]
    groups += [(f"День {k[-1]}", v["by_lead_day"][k]) for k in sorted(v["by_lead_day"])]
    groups += [(f"Турбина {k}", v["by_turbine"][k]) for k in sorted(v["by_turbine"])]
    for name, g in groups:
        mo, pe, cu = g["model"], g["persistence"], g["power_curve"]
        L.append(f"| {name} | Модель | {g['rows']} | {_f(mo['mae'])} | {_f(mo['rmse'])} | "
                 f"{_f(mo['coverage_p10_p90'], 3)} | {_f(mo['mean_width_p10_p90'], 3)} |")
        L.append(f"| {name} | Модель на строках persistence | {pe['rows']} | {_f(mo['mae_on_persistence_rows'])} | "
                 f"{_f(mo['rmse_on_persistence_rows'])} | | |")
        L.append(f"| {name} | Persistence | {pe['rows']} (без факта дня D: {pe['rows_without_fact_for_day_d']}) | "
                 f"{_f(pe['mae'])} | {_f(pe['rmse'])} | | |")
        L.append(f"| {name} | Кривая мощности | {g['rows']} | {_f(cu['mae'])} | {_f(cu['rmse'])} | "
                 f"{_f(cu['coverage_p10_p90'], 3)} | {_f(cu['mean_width_p10_p90'], 3)} |")
    L.append("")
    o = v["overall"]
    L.append(" ".join(x for x in (
        _ratio_sentence(o["model"]["mae_on_persistence_rows"], o["persistence"]["mae"], "persistence"),
        _ratio_sentence(o["model"]["mae"], o["power_curve"]["mae"], "кривой мощности"),
        f"Покрытие коридора модели {_pct(o['model']['coverage_p10_p90'])} при замысле около 80%.") if x) + "\n")
    better = [n for n, g in groups if g["model"]["mae"] is not None and g["power_curve"]["mae"] is not None
              and g["model"]["mae"] < g["power_curve"]["mae"]]
    not_better = [n for n, g in groups if n not in better]
    L.append("Сравнение с кривой мощности по MAE. Модель точнее кривой в срезах: "
             f"{', '.join(better) if better else 'ни в одном'}. Модель не точнее кривой в срезах: "
             f"{', '.join(not_better) if not_better else 'ни в одном'}.\n")
    arch = v.get("archive_forecast_check")
    if arch and arch.get("rows"):
        L.append("Отдельная сверка, чтобы понять, откуда ошибка. Те же модели на тех же часах месяца проверки, "
                 "только признаки взяты из архива прогнозов (как при обучении, самые свежие запуски погодной "
                 f"модели). Строк {arch['rows']}, MAE модели {_f(arch['model_mae'])}, MAE кривой мощности "
                 f"{_f(arch['curve_mae'])}, покрытие коридора модели {_f(arch['model_coverage_p10_p90'], 3)}. "
                 "Разница с таблицей выше показывает, сколько добавляет то, что прогноз погоды сделан за 1–2 "
                 "суток.\n")

    L.append("## Оговорки\n")
    L.append("- Факта за февраль 2026 нет. Проверка сделана на одном месяце, январе 2026. Это зима; как модель "
             "ведёт себя весной и летом, эта проверка не показывает.")
    L.append("- Сервис previous-runs отдаёт прогноз, известный накануне (день 1) и за двое суток (день 2). В какой "
             "час дня выпуска этот запуск погодной модели реально был бы доступен, по данным сервиса не видно.")
    L.append("- Модель обучена на архиве прогнозов (historical-forecast), он склеен из самых свежих запусков "
             "погодной модели. В работе и на проверке модель получает прогноз на 1–2 суток вперёд, он менее "
             "точен. Поэтому на обучении коридор p10–p90 мог выйти уже, чем нужно для прогноза на двое суток. "
             "Покрытие на январе показывает, насколько это заметно.")
    L.append("- Persistence берёт факт за весь день выпуска D. В реальный момент выпуска вторая половина дня D "
             "ещё неизвестна, так что эта точка отсчёта здесь немного сильнее, чем была бы на практике.")
    L.append("- Часы простоя и неполные часы убраны и из обучения, и из проверки. Метрики описывают исправную "
             "турбину. Отключения и ограничения мощности модель не предсказывает.")
    L.append(f"- Сохранённые модели обучены без {VALIDATION_MONTH}. Агент в феврале работает с моделями, "
             "которые январь не видели.")
    L.append("- Погода берётся в одной точке на обе турбины. Различие турбин модель учитывает только через "
             "признак номера турбины.")
    L.append("- Выработка дана в долях номинала; номинальная мощность в данных не указана, поэтому ошибки "
             "в киловаттах здесь не пересчитаны.")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- главный сценарий

def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def train(train_start: str | None = None, artifacts_dir: str | None = None) -> dict:
    """Полный цикл: данные, самопроверка времени, обучение, проверка на месяце VALIDATION_MONTH, артефакты.
    Возвращает метрики (то же, что пишется в metrics.json)."""
    t_start = time.perf_counter()
    train_start = train_start or TRAIN_START
    art = Path(artifacts_dir or ARTIFACTS)
    art.mkdir(parents=True, exist_ok=True)
    _, last_day = _validation_bounds()
    weather_end = last_day.strftime("%Y-%m-%d")
    log.info("Старт обучения: история с %s, месяц проверки %s, артефакты в %s", train_start, VALIDATION_MONTH, art)

    # 1. Факт турбин и фильтр.
    hourly = load_hourly()
    hourly = hourly[hourly["time"] >= pd.Timestamp(train_start, tz=TZ)].reset_index(drop=True)
    if hourly.empty:
        raise ValueError(f"В данных турбин нет часов начиная с {train_start}")
    facts, filter_report = filter_training(hourly)
    data = dict(filter_report)
    for turbine in sorted(TURBINES):
        data[f"hours_total_turbine_{turbine}"] = int((hourly["turbine"] == turbine).sum())
        data.setdefault(f"hours_kept_turbine_{turbine}", 0)
    data["facts_first"] = facts["time"].min().isoformat()
    data["facts_last"] = facts["time"].max().isoformat()

    # 2. Архив прогнозов погоды.
    weather = get_training_weather(train_start, weather_end)
    weather_info = {"hours": int(len(weather)), "first": weather.index.min().isoformat(),
                    "last": weather.index.max().isoformat(), "nan_values": int(weather.isna().sum().sum())}
    log.info("Погода: %d часов, %s … %s, пропусков %d", weather_info["hours"], weather_info["first"],
             weather_info["last"], weather_info["nan_values"])

    # 3. Самопроверка выравнивания по времени.
    alignment = check_alignment(facts, weather)
    applied = alignment["applied_shift_hours"]

    # 4. Склейка и обучение.
    joined = join_features(facts, weather, applied)
    data["rows_without_weather"] = int(len(facts) - len(joined))
    in_val = joined["time"].dt.strftime("%Y-%m") == VALIDATION_MONTH
    train_df = joined[~in_val]
    data["validation_month_rows"] = int(in_val.sum())
    data["train_rows"] = int(len(train_df))
    for turbine in sorted(TURBINES):
        data[f"train_rows_turbine_{turbine}"] = int((train_df["turbine"] == turbine).sum())
    if train_df.empty:
        raise ValueError("Нет обучающих строк: вся история попала в месяц проверки")
    log.info("Обучающих строк %d (турбина 1: %d, турбина 2: %d), строк месяца проверки %d отложено",
             data["train_rows"], data.get("train_rows_turbine_1", 0), data.get("train_rows_turbine_2", 0),
             data["validation_month_rows"])

    X = train_df[FEATURES].to_numpy(dtype=float)
    y = train_df["power"].to_numpy(dtype=float)
    t_fit = time.perf_counter()
    models = {}
    for name, q in QUANTILES.items():
        t0 = time.perf_counter()
        models[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **MODEL_PARAMS).fit(X, y)
        log.info("Модель %s (квантиль %.1f) обучена за %.1f с", name, q, time.perf_counter() - t0)

    # 5. Точка отсчёта: кривая мощности по прогнозному ветру на 100 м, те же обучающие строки.
    curve = PowerCurveModel().fit(train_df["ws100"], train_df["power"], train_df["turbine"])
    train_seconds = time.perf_counter() - t_fit

    for name, model in models.items():
        joblib.dump(model, art / f"model_{name}.joblib")
    joblib.dump(curve, art / "power_curve.joblib")
    log.info("Модели и кривая мощности сохранены в %s", art)

    # 6. Проверка на месяце проверки так, как работает агент.
    rows, val_summary = validate(models, curve, facts, applied)
    # Сверка: те же часы месяца проверки, признаки из архива прогнозов (как на обучении).
    val_df = joined[in_val]
    if len(val_df):
        parts = []
        for turbine, g in val_df.groupby("turbine"):
            f = g.set_index("time")[FEATURES]
            p, c = predict(f, int(turbine), models), curve.predict(f, int(turbine))
            parts.append(pd.DataFrame({"actual": g["power"].to_numpy(), "p10": p["p10"].to_numpy(),
                                       "p50": p["p50"].to_numpy(), "p90": p["p90"].to_numpy(),
                                       "curve": c["p50"].to_numpy()}))
        a = pd.concat(parts, ignore_index=True)
        cov, _ = _band(a["actual"], a["p10"], a["p90"])
        val_summary["archive_forecast_check"] = {
            "rows": int(len(a)), "model_mae": _mae(a["p50"] - a["actual"]),
            "curve_mae": _mae(a["curve"] - a["actual"]), "model_coverage_p10_p90": cov}
        log.info("Сверка на архиве прогнозов за %s: модель MAE %s, кривая MAE %s, покрытие %s",
                 VALIDATION_MONTH, _f(a["p50"].sub(a["actual"]).abs().mean()),
                 _f(a["curve"].sub(a["actual"]).abs().mean()), _f(cov, 3))
    val_summary["overall"] = group_metrics(rows)
    val_summary["by_lead_day"] = {f"day{int(k)}": group_metrics(g) for k, g in rows.groupby("lead_day")}
    val_summary["by_turbine"] = {str(int(k)): group_metrics(g) for k, g in rows.groupby("turbine")}
    for name, g in [("все", val_summary["overall"])] + \
            [(f"день {k[-1]}", g) for k, g in val_summary["by_lead_day"].items()] + \
            [(f"турбина {k}", g) for k, g in val_summary["by_turbine"].items()]:
        log.info("Январь, %s: модель MAE %s RMSE %s покрытие %s ширина %s | persistence MAE %s RMSE %s "
                 "(строк %d) | кривая MAE %s RMSE %s покрытие %s",
                 name, _f(g["model"]["mae"]), _f(g["model"]["rmse"]), _f(g["model"]["coverage_p10_p90"], 3),
                 _f(g["model"]["mean_width_p10_p90"], 3), _f(g["persistence"]["mae"]),
                 _f(g["persistence"]["rmse"]), g["persistence"]["rows"], _f(g["power_curve"]["mae"]),
                 _f(g["power_curve"]["rmse"]), _f(g["power_curve"]["coverage_p10_p90"], 3))

    # 7. Артефакты.
    out = rows.copy()
    out["time"] = out["time"].map(lambda t: t.isoformat())
    out = out[["time", "turbine", "actual", "p10", "p50", "p90", "persistence", "curve", "issue_date", "lead_hours"]]
    out.to_csv(art / "validation.csv", index=False)

    metrics = {
        "train_start": train_start,
        "validation_month": VALIDATION_MONTH,
        "data": data,
        "weather": weather_info,
        "time_alignment": alignment,
        "model": {"type": "HistGradientBoostingRegressor", "loss": "quantile", "quantiles": QUANTILES,
                  "features": FEATURES, "params": MODEL_PARAMS, "train_seconds": train_seconds},
        "validation": val_summary,
    }
    metrics["total_seconds"] = time.perf_counter() - t_start
    metrics = _json_safe(metrics)
    with open(art / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_report(art / "report.md", metrics)
    log.info("Готово за %.1f с. Файлы: %s", metrics["total_seconds"],
             ", ".join(sorted(p.name for p in art.iterdir())))
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    train()


if __name__ == "__main__":
    main()
