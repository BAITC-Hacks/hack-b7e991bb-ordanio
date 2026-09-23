# LLM-аналитик: сводка дня по числам анализа и чат с четырьмя инструментами (прогноз, погода, сравнение
# запусков, проверка модели). Работает только при OPENAI_API_KEY; без ключа сводка шаблонная, чат отключён.

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from common.config import ARTIFACTS, OUTPUT, TEST_ISSUE_DATES

log = logging.getLogger("agent")
load_dotenv()

DEFAULT_MODEL = "gpt-4.1-mini"
NO_KEY_TEXT = ("Чат с агентом отключён: в окружении нет OPENAI_API_KEY. Прогнозы, анализ и журнал "
               "при этом работают полностью; ключ нужен только для ответов на вопросы и сводки дня.")


def llm_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def _client():
    """Клиент OpenAI с коротким таймаутом и без повторов: недоступный API не должен держать цикл агента."""
    from openai import OpenAI
    return OpenAI(timeout=30.0, max_retries=0)


# ---------------------------------------------------------------- сводка дня

def daily_note(analysis: dict) -> str:
    """Сводка дня по-русски. С ключом её пишет модель строго по числам analysis; без ключа или при
    ошибке API возвращается шаблон из тех же чисел."""
    template = template_note(analysis)
    if not llm_enabled():
        return template
    try:
        client = _client()
        prompt = (
            "Ты аналитик ветростанции. Ниже словарь с результатами дневного прогноза выработки двух турбин "
            "на 48 часов (значения p50 в долях номинальной мощности, суммы в единицах «доля × час»). "
            "Напиши сводку дня по-русски, 4–7 предложений, для диспетчера без образования в машинном обучении. "
            "Используй только числа из словаря, ничего не придумывай и не добавляй внешних фактов. "
            "Если поле равно null, скажи, что данных нет. Без заголовков и списков, сплошным текстом.\n\n"
            f"Данные:\n{json.dumps(analysis, ensure_ascii=False)}"
        )
        response = client.chat.completions.create(
            model=_model_name(), temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return template
        return text + "\n\n(Сводку написала модель " + _model_name() + " по числам анализа.)"
    except Exception as exc:
        log.warning("LLM недоступен, сводка по шаблону: %s", exc)
        return template + f"\n\n(LLM недоступен: {exc}; сводка собрана по шаблону.)"


def template_note(analysis: dict) -> str:
    """Шаблонная сводка из чисел analysis, без сети и ключа."""
    days = analysis["days"]
    totals = analysis["totals"]
    w = analysis["weather"]
    parts = []
    parts.append(
        f"Прогноз выпущен {analysis['issue_date']} на {' и '.join(days)}. Ветер на высоте 100 м ожидается "
        f"от {w['ws100_min']} до {w['ws100_max']} м/с, в среднем {w['ws100_mean']}; температура "
        f"от {w['temp_min']} до {w['temp_max']} °C."
    )
    per_t = "; ".join(
        f"турбина {t}: " + ", ".join(f"{d} — {totals[str(t)]['by_day'][d]:.2f}" for d in days)
        + f", всего {totals[str(t)]['total']:.2f}" for t in analysis["turbines"]
    )
    parts.append(f"Ожидаемая выработка (сумма p50, доля номинала × час): {per_t}. "
                 f"Обе турбины за 48 часов: {totals['all']['total']:.2f}.")
    delta = analysis["delta_vs_previous"]
    if delta is None:
        parts.append("Вчерашнего прогноза нет, поэтому сравнение день к дню не проводилось.")
    elif delta.get("hours", 0) == 0:
        parts.append("С вчерашним прогнозом общих часов нет.")
    else:
        direction = "выросла" if delta["delta_energy"] > 0 else ("снизилась" if delta["delta_energy"] < 0 else "не изменилась")
        parts.append(
            f"По сравнению с вчерашним прогнозом на те же {delta['hours']} часов ожидаемая выработка {direction}: "
            f"было {delta['sum_p50_prev']:.2f}, стало {delta['sum_p50_new']:.2f} ({delta['delta_energy']:+.2f}); "
            f"часов, где оценка сдвинулась больше чем на 0.15: {delta['hours_changed_over_threshold']}."
        )
    low = analysis["low_confidence_count"]
    threshold = _threshold_text(analysis)
    if low == 0:
        parts.append(f"Часов низкой уверенности нет ({threshold}).")
    else:
        first = analysis["low_confidence_hours"][0]
        parts.append(f"Часов низкой уверенности: {low} из {analysis['hours'] * len(analysis['turbines'])}, "
                     f"{threshold}; первый из них {first['target_time'][:16]} "
                     f"(турбина {first['turbine']}, {first['reason']}).")
    ext = analysis["extreme_wind_hours"]
    if ext:
        parts.append(f"Часов с ветром выше 25 м/с: {len(ext)}; на эти часы турбины остановлены, "
                     f"выработка принята равной нулю.")
    err = analysis["error_yesterday"]
    if err is None:
        parts.append("Факта за вчера в данных нет, ошибка вчерашнего прогноза не считалась.")
    else:
        parts.append(f"Вчерашний прогноз на {err['day']} разошёлся с фактом в среднем на {err['mae']:.3f} "
                     f"(смещение {err['bias']:+.3f}), факт попал в коридор p10–p90 в {err['coverage_p10_p90'] * 100:.0f}% часов.")
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
        "sum_p50": {d1: round(float(a["p50"].sum()), 3), d2: round(float(b["p50"].sum()), 3)},
        "common_hours": int(len({k[0] for k in common})),
    }
    if len(common):
        diff = (b.loc[common, "p50"] - a.loc[common, "p50"]).astype(float)
        result["common"] = {
            "sum_p50_first": round(float(a.loc[common, "p50"].sum()), 3),
            "sum_p50_second": round(float(b.loc[common, "p50"].sum()), 3),
            "mean_delta_p50": round(float(diff.mean()), 3),
            "mean_abs_delta_p50": round(float(diff.abs().mean()), 3),
            "max_abs_delta_p50": round(float(diff.abs().max()), 3),
            "per_turbine": {
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
    "get_weather": tool_get_weather,
    "compare_runs": tool_compare_runs,
    "get_validation": tool_get_validation,
}

_date_param = {"type": "string", "description": f"Дата выпуска в формате ГГГГ-ММ-ДД, с {TEST_ISSUE_DATES[0]} по {TEST_ISSUE_DATES[1]}"}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Готовый прогноз выработки (p10/p50/p90 по часам и турбинам) и анализ дня для даты выпуска.",
        "parameters": {"type": "object", "properties": {"issue_date": _date_param}, "required": ["issue_date"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Прогноз погоды (ветер 10 и 100 м, порывы, направление, температура, давление) на 48 часов, известный в день выпуска.",
        "parameters": {"type": "object", "properties": {"issue_date": _date_param}, "required": ["issue_date"]},
    }},
    {"type": "function", "function": {
        "name": "compare_runs",
        "description": "Сравнение двух выпусков прогноза по пересекающимся часам.",
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
    "если нужного числа нет, скажи, что данных нет, и не придумывай. Значения p10/p50/p90 — доли "
    "номинальной мощности (0…1), суммы за период — в единицах «доля × час». Прогнозы выпускаются "
    f"на даты с {TEST_ISSUE_DATES[0]} по {TEST_ISSUE_DATES[1]}, каждый на следующие 48 часов."
)

MAX_TOOL_ROUNDS = 6


def ask(question: str, history: list | None = None) -> str:
    """Ответ на вопрос через function calling с четырьмя инструментами. history: список
    {role, content} предыдущих реплик. Любая ошибка API возвращается текстом «LLM недоступен: …»."""
    if not llm_enabled():
        return NO_KEY_TEXT
    if not question or not str(question).strip():
        return "Вопрос пустой. Спросите, например: «Сколько выдаст турбина 1 по прогнозу за 2026-02-05?»"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": str(item["content"])})
    messages.append({"role": "user", "content": str(question).strip()})
    try:
        client = _client()
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=_model_name(), temperature=0, messages=messages, tools=TOOLS_SPEC,
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
