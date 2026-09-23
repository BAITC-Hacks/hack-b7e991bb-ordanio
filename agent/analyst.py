# LLM-аналитик: сводка дня по числам анализа и чат с пятью инструментами (прогноз выпуска, прогноз на
# целевой день, погода, сравнение запусков, проверка модели). Работает только при OPENAI_API_KEY; без ключа сводка шаблонная, чат отключён.

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from common.config import ARTIFACTS, OUTPUT, TEST_ISSUE_DATES

log = logging.getLogger("agent")
load_dotenv()

DEFAULT_MODEL = "gpt-5.5"        # чат (ask), переопределяется OPENAI_MODEL
DEFAULT_NOTE_MODEL = "gpt-5.4"   # сводка дня (make_note), переопределяется OPENAI_MODEL_NOTE
NOTE_TIMEOUT = 45.0              # с, сводка дня
CHAT_TIMEOUT = 30.0              # с, чат
NO_KEY_TEXT = ("Чат с агентом отключён: в окружении нет OPENAI_API_KEY. Прогнозы, анализ и журнал "
               "при этом работают полностью; ключ нужен только для ответов на вопросы и сводки дня.")


def llm_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def _note_model_name() -> str:
    """Модель сводки дня: OPENAI_MODEL_NOTE, иначе DEFAULT_NOTE_MODEL (чат идёт на _model_name())."""
    return os.environ.get("OPENAI_MODEL_NOTE", "").strip() or DEFAULT_NOTE_MODEL


def _client(timeout: float = CHAT_TIMEOUT):
    """Клиент OpenAI с таймаутом и без повторов: недоступный API не должен держать цикл агента.
    Чат 30 с (CHAT_TIMEOUT), сводка дня 45 с (NOTE_TIMEOUT); при превышении сводка собирается по шаблону."""
    from openai import OpenAI
    return OpenAI(timeout=timeout, max_retries=0)


# ---------------------------------------------------------------- сводка дня

def daily_note(analysis: dict) -> str:
    """Сводка дня по-русски (контракт): текст из make_note в режиме "auto"."""
    return make_note(analysis, "auto")[0]


def make_note(analysis: dict, mode: str = "auto") -> tuple[str, str]:
    """Сводка дня и её источник: (text, "model:<имя модели>") или (text, "template").
    mode="auto": модель при ключе, иначе шаблон; mode="template": всегда шаблон, без сети.
    При ошибке или таймауте API возвращается шаблон с припиской, источник "template"."""
    template = template_note(analysis)
    if mode == "template" or not llm_enabled():
        return template, "template"
    if mode != "auto":
        raise ValueError(f"Неизвестный режим сводки «{mode}»: допустимы auto и template.")
    try:
        client = _client(NOTE_TIMEOUT)
        model = _note_model_name()
        prompt = (
            "Ты аналитик ветростанции из двух турбин. Ниже словарь с результатами дневного прогноза выработки "
            "на 48 часов. Напиши сводку дня по-русски для диспетчера без образования в машинном обучении.\n"
            "Правила:\n"
            "- Сплошной текст, 4–7 предложений, без заголовков, списков и markdown (никаких звёздочек и решёток).\n"
            "- Используй только числа из словаря, ничего не придумывай и не добавляй внешних фактов.\n"
            "- Дробные величины (мощность, суммы, ветер, температура, пороги) округляй до двух знаков после запятой "
            "(2.774 пиши как 2.77). Количества часов и дней пиши целыми, без дробной части (13 часов, а не 13.00).\n"
            "- Единицы называй словами: p10/p50/p90 и пики — «доля номинальной мощности» (от 0 до 1); суммы "
            "за период — «сумма нормализованной мощности по часам, доля номинала × час»; ветер — м/с; температура — °C. "
            "Суммы p10 и p90 по часам — это сумма квантилей, а не интервал суточной энергии; так их и не называй.\n"
            "- Поле weather_lead_check: перескажи одной фразой его text (проверка по номинальному упреждению, что вся "
            "погода рассчитана не позже момента выпуска; называй её именно «по номинальному упреждению»); если ok "
            "равно false, прямо скажи, что по номинальному упреждению проверка не пройдена и почему.\n"
            "- Счётчики low_confidence_count, hours_changed_over_threshold и cutout_rows считают строки «час × турбина» "
            "(до 96 за выпуск), а не часы: называй их «пар час–турбина», не «часов». Часами называй только hours, "
            "delta_vs_previous.hours и cutout_hours.\n"
            "- Если есть поле previous_forecast_warning, перескажи его: вчерашний прогноз не прочитан, сравнивать не с чем.\n"
            "- Поле cutout_hours — часы предполагаемой остановки (ветер выше 25 м/с, порог не подтверждён "
            "паспортом турбин): все квантили там приняты равными 0, уверенность low. Если cutout_hours больше 0, "
            "назови их число этими словами; если 0, напиши «Часов предполагаемой остановки: нет». Не выдавай "
            "остановку за известный факт.\n"
            "- Время и даты пиши по-человечески: «13 февраля в 22:00», «13 февраля», без формата ISO, без буквы T "
            "и без часового смещения.\n"
            "- Не упоминай служебные имена полей (issue_date, totals, delta_vs_previous, error_yesterday, "
            "low_confidence, weather_source, cache, network и подобные) — переводи их в смысл. Источник погоды: "
            "cache пиши как «архив прогнозов из кэша», network — как «Open-Meteo».\n"
            "- delta_vs_previous — это сравнение с прогнозом, выпущенным днём раньше, на те же часы; если он null, "
            "напиши, что вчерашнего прогноза для сравнения нет.\n"
            "- Если error_yesterday равно null, напиши «факта за вчера нет»; иначе назови ошибку вчерашнего "
            "прогноза по факту.\n"
            "- Про пары час–турбина низкой уверенности назови их число и порог; порог словами: "
            f"{_threshold_text(analysis)}.\n\n"
            f"Данные:\n{json.dumps(analysis, ensure_ascii=False)}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return template, "template"
        return (text + "\n\n(Сводку написала модель " + model + " по числам анализа.)", "model:" + model)
    except Exception as exc:
        log.warning("LLM недоступен, сводка по шаблону: %s", exc)
        return template + f"\n\n(LLM недоступен: {exc}; сводка собрана по шаблону.)", "template"


def template_note(analysis: dict) -> str:
    """Шаблонная сводка из чисел analysis, без сети и ключа. Даты словами, суммы с двумя знаками,
    ветер и температура с одним (исходная точность прогноза погоды), без markdown."""
    from agent.tools import _hours_word, _ru_day, _ru_time, cutout_text
    days = analysis["days"]
    totals = analysis["totals"]
    w = analysis["weather"]
    parts = []
    parts.append(
        f"Прогноз выпущен {_ru_day(analysis['issue_date'])} на {' и '.join(_ru_day(d) for d in days)}. "
        f"Ветер на высоте 100 м ожидается от {w['ws100_min']:.1f} до {w['ws100_max']:.1f} м/с, в среднем "
        f"{w['ws100_mean']:.1f} м/с; температура от {w['temp_min']:.1f} до {w['temp_max']:.1f} °C."
    )
    check = analysis.get("weather_lead_check")
    if check:  # в журнале эта фраза уже стоит отдельной строкой; здесь она внутри предложения
        text = str(check["text"])
        parts.append("Момент расчёта погоды: " + text[:1].lower() + text[1:] + ".")
    per_t = "; ".join(
        f"турбина {t}: " + ", ".join(f"{_ru_day(d)} — {totals[str(t)]['by_day'][d]:.2f}" for d in days)
        + f", всего {totals[str(t)]['total']:.2f}" for t in analysis["turbines"]
    )
    total_hours = analysis["hours"]
    parts.append(f"Ожидаемая выработка (сумма p50, доля номинала × час): {per_t}. "
                 f"Обе турбины за {total_hours} {_hours_word(total_hours)}: {totals['all']['total']:.2f}. "
                 f"Все суммы здесь — сумма нормализованной мощности по часам (доля номинала × час); суммы p10 и p90 "
                 f"по часам — это сумма квантилей, а не интервал суточной энергии.")
    delta = analysis["delta_vs_previous"]
    warning = analysis.get("previous_forecast_warning")
    if delta is None and warning:
        parts.append(warning[:1].upper() + warning[1:] + ".")
    elif delta is None:
        parts.append("Вчерашнего прогноза нет, поэтому сравнение день к дню не проводилось.")
    elif delta.get("hours", 0) == 0:
        parts.append("С вчерашним прогнозом общих часов нет.")
    else:
        direction = "выросла" if delta["delta_energy"] > 0 else ("снизилась" if delta["delta_energy"] < 0 else "не изменилась")
        changed = delta["hours_changed_over_threshold"]
        parts.append(
            f"По сравнению с вчерашним прогнозом на те же {delta['hours']} {_hours_word(delta['hours'])} ожидаемая "
            f"выработка {direction}: было {delta['sum_p50_prev']:.2f}, стало {delta['sum_p50_new']:.2f} "
            f"({delta['delta_energy']:+.2f} доли номинала × час); пар час–турбина, где оценка сдвинулась больше чем "
            f"на 0.15: {changed}."
        )
    low = analysis["low_confidence_count"]
    threshold = _threshold_text(analysis)
    if low == 0:
        parts.append(f"Пар час–турбина низкой уверенности нет ({threshold}).")
    else:
        first = analysis["low_confidence_hours"][0]
        parts.append(f"Пар час–турбина низкой уверенности: {low} из {analysis['hours'] * len(analysis['turbines'])}, "
                     f"{threshold}; первая из них {_ru_time(first['target_time'], sep=" в ")} "
                     f"(турбина {first['turbine']}, {first['reason']}).")
    cutout = analysis.get("cutout_hours", len(analysis.get("extreme_wind_hours") or []))
    if cutout:
        parts.append(cutout_text(analysis))
    err = analysis["error_yesterday"]
    if err is None:
        parts.append("Факта за вчера нет, ошибка вчерашнего прогноза не считалась.")
    else:
        parts.append(f"Вчерашний прогноз на {_ru_day(err['day'])} разошёлся с фактом в среднем на {err['mae']:.2f} "
                     f"доли номинальной мощности (смещение {err['bias']:+.2f}), факт попал в коридор p10–p90 "
                     f"в {err['coverage_p10_p90'] * 100:.0f}% часов.")
    return " ".join(parts)


def _threshold_text(analysis: dict) -> str:
    """Порог низкой уверенности словами; текст общий с журналом (agent.tools.threshold_text)."""
    from agent.tools import threshold_text
    return threshold_text(analysis)


# ---------------------------------------------------------------- чат с инструментами

def tool_get_forecast(issue_date: str) -> dict:
    """Строки прогноза из output/forecasts/forecast_<issue_date>.csv, если файл есть."""
    import pandas as pd
    path = os.path.join(OUTPUT, "forecasts", f"forecast_{issue_date}.csv")
    if not os.path.exists(path):
        return {"error": f"Прогноза за {issue_date} нет: файл {path} не найден. Сначала сформируйте прогноз."}
    rows = pd.read_csv(path, keep_default_na=False)
    summary = {}
    for t, sub in rows.groupby("turbine"):
        summary[str(int(t))] = {
            "sum_p50": round(float(sub["p50"].sum()), 3),
            "sum_p10": round(float(sub["p10"].sum()), 3),
            "sum_p90": round(float(sub["p90"].sum()), 3),
            "max_p50": round(float(sub["p50"].max()), 3),
            "max_p50_time": str(sub.loc[sub["p50"].idxmax(), "target_time"]),
            "low_confidence_hours": int((sub["confidence"] == "low").sum()),
        }
    run_path = os.path.join(OUTPUT, "runs", f"{issue_date}.json")
    analysis = None
    if os.path.exists(run_path):
        with open(run_path, encoding="utf-8") as f:
            analysis = json.load(f).get("analysis")
    return {"issue_date": issue_date, "rows": len(rows), "summary_by_turbine": summary,
            "analysis": analysis, "hours": rows.to_dict(orient="records")}


def tool_get_forecast_for_day(target_date: str) -> dict:
    """Самый свежий прогноз на целевой день выработки target_date. Выпуск D покрывает дни D+1 и D+2,
    поэтому сначала ищется forecast_<target−1>.csv (лаг «завтра»), затем forecast_<target−2>.csv
    («послезавтра»). Возвращает часы этого дня (48 строк: 24 часа × 2 турбины) и сводку по турбинам."""
    import pandas as pd
    from datetime import date, timedelta
    try:
        target = date.fromisoformat(str(target_date).strip())
    except ValueError:
        return {"error": f"Не понял дату «{target_date}»: нужен формат ГГГГ-ММ-ДД, например 2026-02-13."}
    target_str = target.isoformat()
    tried = []
    for lag, lead in ((1, "завтра"), (2, "послезавтра")):
        issue = (target - timedelta(days=lag)).isoformat()
        path = os.path.join(OUTPUT, "forecasts", f"forecast_{issue}.csv")
        tried.append(path)
        if not os.path.exists(path):
            continue
        rows = pd.read_csv(path, keep_default_na=False)
        day_rows = rows[rows["target_time"].astype(str).str[:10] == target_str]
        if day_rows.empty:
            continue
        summary = {}
        for t, sub in day_rows.groupby("turbine"):
            idx = sub["p50"].astype(float).idxmax()
            summary[str(int(t))] = {
                "sum_p50": round(float(sub["p50"].sum()), 3),
                "sum_p10": round(float(sub["p10"].sum()), 3),
                "sum_p90": round(float(sub["p90"].sum()), 3),
                "max_p50": round(float(sub["p50"].max()), 3),
                "max_p50_time": str(sub.loc[idx, "target_time"]),
                "low_confidence_hours": int((sub["confidence"] == "low").sum()),
                "hours": int(len(sub)),
            }
        return {
            "target_date": target_str, "issue_date": issue, "lead": lead,
            "source_file": path, "rows": int(len(day_rows)),
            "summary_by_turbine": summary,
            "hours": day_rows.to_dict(orient="records"),
        }
    return {"error": (f"Прогноза на {target_str} нет: искал файлы {tried[0]} (выпуск "
                      f"{(target - timedelta(days=1)).isoformat()}) и {tried[1]} (выпуск "
                      f"{(target - timedelta(days=2)).isoformat()}), ни один не найден или не покрывает этот день. "
                      f"Сначала сформируйте прогноз за день выпуска {(target - timedelta(days=1)).isoformat()}.")}


def tool_get_weather(issue_date: str) -> dict:
    """Прогноз погоды, известный в день issue_date, на 48 часов (через model.weather, кэш или сеть)."""
    try:
        from model.weather import get_issued_forecast
        weather = get_issued_forecast(issue_date)
    except Exception as exc:
        return {"error": f"Погоду за {issue_date} получить не удалось: {exc}"}
    out = weather.reset_index()
    out.columns = [str(c) for c in out.columns]
    time_col = out.columns[0]
    out[time_col] = out[time_col].astype(str)
    rounded = out.round(2)
    return {
        "issue_date": issue_date, "hours": len(rounded),
        "ws100_min": round(float(weather["ws100"].min()), 1),
        "ws100_max": round(float(weather["ws100"].max()), 1),
        "ws100_mean": round(float(weather["ws100"].mean()), 1),
        "rows": rounded.to_dict(orient="records"),
    }


def tool_compare_runs(d1: str, d2: str) -> dict:
    """Сравнение двух выпусков по пересекающимся часам: сумма p50, средняя разница по турбинам."""
    import pandas as pd
    paths = {d: os.path.join(OUTPUT, "forecasts", f"forecast_{d}.csv") for d in (d1, d2)}
    missing = [d for d, p in paths.items() if not os.path.exists(p)]
    if missing:
        return {"error": f"Нет прогноза за {', '.join(missing)}. Сначала сформируйте эти дни."}
    a = pd.read_csv(paths[d1], keep_default_na=False).set_index(["target_time", "turbine"])
    b = pd.read_csv(paths[d2], keep_default_na=False).set_index(["target_time", "turbine"])
    common = a.index.intersection(b.index)
    result = {
        "issue_dates": [d1, d2],
        "full_run_sum_p50_48h": {d1: round(float(a["p50"].sum()), 3), d2: round(float(b["p50"].sum()), 3)},
        "full_run_note": ("full_run_sum_p50_48h: сумма p50 обеих турбин за все 48 часов каждого выпуска; периоды "
                          "у выпусков разные, поэтому эти две суммы между собой напрямую не сравниваются. "
                          "Сравнение выпусков — только по блоку common (одни и те же часы)."),
        "common_hours": int(len({k[0] for k in common})),
    }
    if len(common):
        diff = (b.loc[common, "p50"] - a.loc[common, "p50"]).astype(float)
        times = sorted({str(k[0]) for k in common})
        result["common"] = {
            "period": [times[0], times[-1]],
            "sum_p50_both_turbines_first": round(float(a.loc[common, "p50"].sum()), 3),
            "sum_p50_both_turbines_second": round(float(b.loc[common, "p50"].sum()), 3),
            "sum_p50_by_turbine": {
                str(int(t)): {d1: round(float(a.loc[[k for k in common if k[1] == t], "p50"].sum()), 3),
                              d2: round(float(b.loc[[k for k in common if k[1] == t], "p50"].sum()), 3)}
                for t in sorted({k[1] for k in common})
            },
            "mean_delta_p50": round(float(diff.mean()), 3),
            "mean_abs_delta_p50": round(float(diff.abs().mean()), 3),
            "max_abs_delta_p50": round(float(diff.abs().max()), 3),
            "mean_delta_p50_by_turbine": {
                str(int(t)): round(float(diff[[k for k in common if k[1] == t]].mean()), 3)
                for t in sorted({k[1] for k in common})
            },
        }
    return result


def tool_get_validation() -> dict:
    """Метрики модели на январе 2026 из model/artifacts/metrics.json и итог по validation.csv."""
    import pandas as pd
    metrics_path = os.path.join(ARTIFACTS, "metrics.json")
    if not os.path.exists(metrics_path):
        return {"error": f"Метрик нет: файл {metrics_path} не найден. Сначала обучите модель (python -m model.train)."}
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    result = {"metrics": metrics}
    val_path = os.path.join(ARTIFACTS, "validation.csv")
    if os.path.exists(val_path):
        val = pd.read_csv(val_path)
        result["validation_rows"] = int(len(val))
        if {"actual", "p50"} <= set(val.columns):
            err = (val["p50"] - val["actual"]).astype(float)
            result["validation_summary"] = {
                "mae_p50": round(float(err.abs().mean()), 4),
                "rmse_p50": round(float((err ** 2).mean() ** 0.5), 4),
                "mean_actual": round(float(val["actual"].mean()), 4),
                "mean_p50": round(float(val["p50"].mean()), 4),
            }
    return result


TOOL_FUNCTIONS = {
    "get_forecast": tool_get_forecast,
    "get_forecast_for_day": tool_get_forecast_for_day,
    "get_weather": tool_get_weather,
    "compare_runs": tool_compare_runs,
    "get_validation": tool_get_validation,
}

_ISSUE_NOTE = ("issue_date это день выпуска прогноза; прогноз покрывает два следующих дня: issue_date+1 "
               "и issue_date+2. Для вопроса про конкретный день выработки нужен get_forecast_for_day.")
_SUM_NOTE = ("Суммы p10 и p90 за период это суммы квантилей по часам, а не интервал суточной выработки; "
             "не подавать их как «от X до Y».")
_date_param = {"type": "string", "description": (
    f"День выпуска прогноза (issue_date) в формате ГГГГ-ММ-ДД, с {TEST_ISSUE_DATES[0]} по {TEST_ISSUE_DATES[1]}. "
    "Это НЕ день выработки: прогноз покрывает два следующих дня, issue_date+1 и issue_date+2.")}
_target_param = {"type": "string", "description": (
    "Целевой день выработки в формате ГГГГ-ММ-ДД, тот день, который назвал пользователь "
    "(например, «13 февраля» это 2026-02-13).")}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": ("Готовый прогноз выработки (p10/p50/p90 по часам и турбинам) и анализ одного выпуска "
                        "целиком, 48 часов. " + _ISSUE_NOTE + " " + _SUM_NOTE),
        "parameters": {"type": "object", "properties": {"issue_date": _date_param}, "required": ["issue_date"]},
    }},
    {"type": "function", "function": {
        "name": "get_forecast_for_day",
        "description": ("Прогноз выработки на один целевой день (24 часа × 2 турбины) из самого свежего выпуска, "
                        "который этот день покрывает: сначала выпуск target_date−1, если его нет, то target_date−2. "
                        "Возвращает, из какого выпуска взят прогноз (issue_date, lead «завтра»/«послезавтра»), "
                        "часы дня и сводку по турбинам: sum_p50, sum_p10, sum_p90, max_p50 и его час, "
                        "число часов низкой уверенности. Звать всегда, когда пользователь спрашивает про конкретный день. "
                        + _SUM_NOTE),
        "parameters": {"type": "object", "properties": {"target_date": _target_param}, "required": ["target_date"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": ("Прогноз погоды (ветер 10 и 100 м, порывы, направление, температура, давление) на 48 часов, "
                        "известный в день выпуска. " + _ISSUE_NOTE),
        "parameters": {"type": "object", "properties": {"issue_date": _date_param}, "required": ["issue_date"]},
    }},
    {"type": "function", "function": {
        "name": "compare_runs",
        "description": ("Сравнение двух выпусков прогноза по пересекающимся часам (соседние выпуски пересекаются "
                        "на одних сутках). d1 и d2 это дни выпуска; " + _ISSUE_NOTE),
        "parameters": {"type": "object", "properties": {"d1": _date_param, "d2": _date_param}, "required": ["d1", "d2"]},
    }},
    {"type": "function", "function": {
        "name": "get_validation",
        "description": "Качество модели на январе 2026: MAE, RMSE, покрытие коридора, точки отсчёта.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

SYSTEM_PROMPT = (
    "Ты аналитик ветростанции из двух турбин. Отвечай по-русски, коротко и по делу, для диспетчера "
    "без образования в машинном обучении. Отвечай только по данным, которые вернули инструменты: "
    "если нужного числа нет, скажи, что данных нет, и не придумывай. "
    "Даты. Прогноз выпускается в день issue_date (выпуски с "
    f"{TEST_ISSUE_DATES[0]} по {TEST_ISSUE_DATES[1]}, год 2026) и покрывает два следующих дня: issue_date+1 "
    "и issue_date+2. Если пользователь называет день («выработка 13 февраля», «что будет 20-го»), он имеет "
    "в виду целевой день выработки, а не день выпуска: для такого вопроса вызывай get_forecast_for_day "
    "с target_date этого дня. get_forecast, get_weather и compare_runs принимают день выпуска; "
    "звать их, только если пользователь прямо говорит о выпуске или о 48 часах выпуска. "
    "В ответе всегда называй, из какого выпуска взят прогноз (дата выпуска и «на завтра»/«на послезавтра»). "
    "Сегодня. Если пользователь говорит «сегодня/завтра/послезавтра» и не называет дату, считай сегодняшним днём "
    f"последний выпуск {TEST_ISSUE_DATES[1]} и назови это допущение в ответе. "
    "Суммы p10 и p90 за период это суммы квантилей по часам, а не интервал суточной выработки; не подавай их "
    "как «от X до Y». "
    "Любое число из инструментов округляй до двух знаков после запятой (например, -0.075 пиши как -0.08; 0.6 как 0.60); количества часов и дней — целыми числами. Называй единицы: p10/p50/p90 — доля номинальной мощности "
    "(от 0 до 1); суммы за период — в единицах «доля номинала × час». Давай числа по каждой турбине. "
    "Пиши без markdown-разметки: без звёздочек, решёток и заголовков; списки допустимы только "
    "простыми строками, начинающимися с дефиса."
)

MAX_TOOL_ROUNDS = 6


def ask(question: str, history: list | None = None) -> str:
    """Ответ на вопрос через function calling с пятью инструментами. history: список
    {role, content} предыдущих реплик. Любая ошибка API возвращается текстом «LLM недоступен: …»."""
    if not llm_enabled():
        return NO_KEY_TEXT
    if not question or not str(question).strip():
        return "Вопрос пустой. Спросите, например: «Сколько выдаст турбина 1 по прогнозу за 2026-02-05?»"
    try:
        # history любого вида (None, число, строка, список с мусором) не роняет чат: берём только
        # словари {role: user|assistant, content}, остальное пропускаем.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        items = history if isinstance(history, (list, tuple)) else []
        for item in items:
            if isinstance(item, dict) and item.get("role") in ("user", "assistant") and item.get("content"):
                messages.append({"role": item["role"], "content": str(item["content"])})
        messages.append({"role": "user", "content": str(question).strip()})
        client = _client(CHAT_TIMEOUT)
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=_model_name(), messages=messages, tools=TOOLS_SPEC,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return (message.content or "").strip() or "Модель вернула пустой ответ."
            messages.append({"role": "assistant", "content": message.content or "",
                             "tool_calls": [tc.model_dump() for tc in message.tool_calls]})
            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                log.info("Чат: инструмент %s(%s)", name, args)
                fn = TOOL_FUNCTIONS.get(name)
                if fn is None:
                    payload = {"error": f"Неизвестный инструмент {name}"}
                else:
                    try:
                        payload = fn(**args)
                    except TypeError as exc:
                        payload = {"error": f"Неверные аргументы для {name}: {exc}"}
                    except Exception as exc:
                        payload = {"error": f"Инструмент {name} завершился с ошибкой: {exc}"}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(_trim(payload), ensure_ascii=False)})
        return "Не удалось собрать ответ: модель слишком долго вызывала инструменты. Уточните вопрос."
    except Exception as exc:
        log.warning("LLM недоступен: %s", exc)
        return f"LLM недоступен: {exc}"


def _trim(payload: dict, max_rows: int = 96) -> dict:
    """Урезает длинные списки строк, чтобы не раздувать контекст модели."""
    if isinstance(payload, dict):
        for key in ("rows", "hours"):
            if isinstance(payload.get(key), list) and len(payload[key]) > max_rows:
                payload[key] = payload[key][:max_rows]
                payload[key + "_truncated"] = True
    return payload
