# API и страница проекта: запуск суточного цикла агента, выдача прогнозов, журнала, проверки модели и чата.
# Запуск: python -m web.app (слушает 0.0.0.0:8000). Пути относительные от корня репозитория.

import dataclasses
import datetime as dt
import importlib
import json
import logging
import math
import os
import runpy
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from common.config import ARTIFACTS, OUTPUT, TEST_ISSUE_DATES, WEATHER_CACHE

load_dotenv()

log = logging.getLogger("web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path("web") / "static"
FORECASTS_DIR = Path(OUTPUT) / "forecasts"
JOURNAL_PATH = Path(OUTPUT) / "journal.md"
ARTIFACT_FILES = ("model_q10.joblib", "model_q50.joblib", "model_q90.joblib", "metrics.json")

# Состояние фонового обучения: идёт ли, чем закончилось.
training_state = {"running": False, "error": None, "started": None, "finished": None}
# Один запуск агента за раз: журнал и файлы пишутся последовательно.
run_lock = threading.Lock()


# ---------- вспомогательные функции ----------

def artifacts_ready() -> bool:
    """Все файлы модели на месте."""
    return all((Path(ARTIFACTS) / name).exists() for name in ARTIFACT_FILES)


def weather_cache_ok() -> bool:
    """В кэше погоды есть хотя бы один ответ Open-Meteo."""
    cache = Path(WEATHER_CACHE)
    return cache.is_dir() and any(cache.glob("*.json"))


def llm_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def offline() -> bool:
    return os.environ.get("WEATHER_OFFLINE", "0").strip() == "1"


def module_or_503(name: str):
    """Импортирует модуль другого прораба. Пока его нет, отвечает 503 понятным текстом."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == name or name.startswith(exc.name + ".") or exc.name.startswith(name.split(".")[0])):
            raise HTTPException(503, f"Модуль «{name}» ещё не готов: {exc}. Страница работает, запуск пока недоступен.")
        raise HTTPException(503, f"Модуль «{name}» не загружается, не хватает зависимости: {exc}")
    except Exception as exc:  # ошибка внутри чужого модуля
        raise HTTPException(503, f"Модуль «{name}» не загружается: {type(exc).__name__}: {exc}")


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def parse_issue_date(value: str | None, field: str = "issue_date") -> str:
    """Проверяет дату выпуска: формат ГГГГ-ММ-ДД и попадание в тестовый период. Иначе 400 по-русски."""
    lo, hi = TEST_ISSUE_DATES
    if value is None or not str(value).strip():
        raise HTTPException(400, f"Не указана дата выпуска прогноза ({field}). Ожидается дата от {lo} до {hi}.")
    value = str(value).strip()
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        raise HTTPException(400, f"«{value}» не похоже на дату. Нужен формат ГГГГ-ММ-ДД, например {lo}.")
    text = parsed.isoformat()
    if not (lo <= text <= hi):
        raise HTTPException(400, f"Дата {text} вне тестового периода. Допустимы даты от {lo} до {hi}.")
    return text


def jsonable(obj):
    """Переводит результат агента (dataclass, pandas, numpy, даты) в то, что можно отдать как JSON."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy-скаляр
        return jsonable(obj.item())
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return jsonable(vars(obj))
    return str(obj)


def forecast_path(issue_date: str) -> Path:
    return FORECASTS_DIR / f"forecast_{issue_date}.csv"


def read_forecast_rows(issue_date: str) -> list[dict]:
    """Строки файла прогноза как список словарей; пустые ячейки становятся null."""
    path = forecast_path(issue_date)
    if not path.exists():
        raise HTTPException(404, f"Прогноза на дату выпуска {issue_date} ещё нет. Сначала сформируйте его.")
    import pandas as pd

    df = pd.read_csv(path)
    return jsonable(df.astype(object).where(df.notna(), None).to_dict("records"))


def train_in_background(reason: str):
    """Запускает обучение модели в отдельном потоке, если модуль model.train есть."""
    if training_state["running"]:
        return
    if not module_available("model.train"):
        log.warning("Артефактов модели нет, а модуль model.train ещё не готов: обучение не запущено.")
        return

    def worker():
        training_state.update(running=True, error=None, started=time.time(), finished=None)
        log.info("Артефактов модели нет (%s). Запускаю обучение в фоне, погода берётся из кэша.", reason)
        try:
            train_mod = importlib.import_module("model.train")
            entry = getattr(train_mod, "main", None) or getattr(train_mod, "train", None)
            if callable(entry):
                entry()
            else:
                runpy.run_module("model.train", run_name="__main__")
            log.info("Обучение завершено, артефакты в %s.", ARTIFACTS)
        except Exception as exc:
            training_state["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("Обучение упало: %s", exc)
        finally:
            training_state.update(running=False, finished=time.time())

    threading.Thread(target=worker, name="model-train", daemon=True).start()


# ---------- приложение ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    if artifacts_ready():
        log.info("Артефакты модели найдены в %s.", ARTIFACTS)
    else:
        train_in_background("при старте сервера")
    log.info("Ключ OpenAI: %s. Режим без сети: %s.", "есть" if llm_enabled() else "нет, чат и сводка отключены",
             "да" if offline() else "нет")
    yield


app = FastAPI(title="Прогноз выработки ВЭС", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Ошибка при обработке %s: %s", request.url.path, exc)
    return JSONResponse(500, {"detail": f"Внутренняя ошибка: {type(exc).__name__}: {exc}"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    ready = artifacts_ready()
    if not ready and not training_state["running"] and not training_state["error"]:
        train_in_background("по запросу статуса")  # модуль мог появиться после старта
    return {
        "model_ready": ready,
        "weather_cache_ok": weather_cache_ok(),
        "llm_enabled": llm_enabled(),
        "offline": offline(),
        "training": training_state["running"],
        "training_error": training_state["error"],
        "modules": {"agent": module_available("agent.run"), "model": module_available("model.predict")},
        "period": list(TEST_ISSUE_DATES),
        "forecasts_done": sorted(p.stem.replace("forecast_", "") for p in FORECASTS_DIR.glob("forecast_*.csv")),
    }


def require_runnable():
    if training_state["running"]:
        raise HTTPException(503, "Модель ещё обучается. Подождите, статус обновится сам.")


@app.post("/api/run")
def run(issue_date: str | None = Query(None)):
    issue_date = parse_issue_date(issue_date)
    require_runnable()
    agent_run = module_or_503("agent.run")
    with run_lock:
        try:
            result = agent_run.run_day(issue_date)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("run_day(%s) упал", issue_date)
            raise HTTPException(500, f"Агент не смог сформировать прогноз на {issue_date}: {type(exc).__name__}: {exc}")
    payload = jsonable(result)
    if not isinstance(payload, dict):
        payload = {"result": payload}
    path = payload.get("forecast_path") or str(forecast_path(issue_date))
    payload["rows"] = read_forecast_rows(issue_date) if forecast_path(issue_date).exists() else []
    payload["forecast_path"] = path
    return payload


@app.post("/api/run-period")
def run_period(from_: str | None = Query(None, alias="from"), to: str | None = Query(None)):
    start = parse_issue_date(from_, "from")
    end = parse_issue_date(to, "to")
    if start > end:
        raise HTTPException(400, f"Начало периода {start} позже конца {end}.")
    require_runnable()
    agent_run = module_or_503("agent.run")
    with run_lock:
        try:
            results = agent_run.run_period(start, end)
        except Exception as exc:
            log.exception("run_period(%s, %s) упал", start, end)
            raise HTTPException(500, f"Прогон периода {start}…{end} прерван: {type(exc).__name__}: {exc}")
    return jsonable(list(results))


@app.get("/api/forecast/{issue_date}.csv")
def forecast_csv(issue_date: str):
    issue_date = parse_issue_date(issue_date)
    path = forecast_path(issue_date)
    if not path.exists():
        raise HTTPException(404, f"Прогноза на дату выпуска {issue_date} ещё нет. Сначала сформируйте его.")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/forecast/{issue_date}")
def forecast_json(issue_date: str):
    issue_date = parse_issue_date(issue_date)
    return {"issue_date": issue_date, "rows": read_forecast_rows(issue_date)}


@app.get("/api/journal", response_class=PlainTextResponse)
def journal():
    if not JOURNAL_PATH.exists():
        return "Журнал пока пуст: агент ещё не делал ни одного прогноза."
    return JOURNAL_PATH.read_text(encoding="utf-8")


@app.get("/api/validation")
def validation():
    metrics_path = Path(ARTIFACTS) / "metrics.json"
    csv_path = Path(ARTIFACTS) / "validation.csv"
    if not metrics_path.exists():
        if training_state["running"]:
            raise HTTPException(503, "Модель ещё обучается, метрики появятся после обучения.")
        raise HTTPException(503, "Метрик пока нет: модель не обучена. Запустите python -m model.train.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        rows = jsonable(df.astype(object).where(df.notna(), None).to_dict("records"))
    report_path = Path(ARTIFACTS) / "report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {"metrics": metrics, "rows": rows, "report": report}


@app.post("/api/ask")
async def ask(request: Request):
    if not llm_enabled():
        raise HTTPException(503, "Чат с агентом отключён: не задан OPENAI_API_KEY в .env.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Ожидается JSON вида {\"question\": \"...\", \"history\": []}.")
    question = str(body.get("question", "")).strip() if isinstance(body, dict) else ""
    if not question:
        raise HTTPException(400, "Вопрос пустой. Напишите, что хотите узнать о прогнозе.")
    history = body.get("history") or []
    analyst = module_or_503("agent.analyst")
    try:
        answer = analyst.ask(question, history)
    except Exception as exc:
        log.exception("ask() упал")
        raise HTTPException(500, f"Агент не смог ответить: {type(exc).__name__}: {exc}")
    return {"answer": answer}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn

    # Все пути в проекте относительные, поэтому сервер работает из корня репозитория.
    os.chdir(Path(__file__).resolve().parent.parent)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
