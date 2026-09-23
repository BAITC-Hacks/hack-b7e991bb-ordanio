# Архитектура и контракт модулей

Три части строятся параллельно и стыкуются по этому контракту. Имена папок, файлов, функций
и колонок ниже обязательны: менять их можно только вместе с этим документом.

## Общие константы: `common/config.py`

```python
TURBINES = {1: (43.645150, 78.535604), 2: (43.643198, 78.538828)}  # широта, долгота из ссылок ТЗ
WEATHER_POINT = (43.6442, 78.5372)   # одна точка на обе турбины: 240 м друг от друга, одна ячейка сетки
TZ = "Asia/Almaty"                   # местное время данных; перевод на UTC+5 в марте 2024 учтён в tz-базе
DATA_RAW = "data/raw"                # turbine_1.csv, turbine_2.csv
WEATHER_CACHE = "data/weather_cache" # json-ответы Open-Meteo
ARTIFACTS = "model/artifacts"        # модели, метрики, отчёт
OUTPUT = "output"                    # forecasts/, runs/, journal.md
TRAIN_START = "2023-03-11"
VALIDATION_MONTH = "2026-01"         # в обучение не входит
TEST_ISSUE_DATES = ("2026-01-31", "2026-02-28")  # даты выпуска прогноза
HORIZON_HOURS = 48
```

Все пути относительные от корня репозитория. В коде не бывает `/Users/...` и shell-команд.
Время везде: `pandas.Timestamp` с tz `Asia/Almaty`, шаг час.

## Погода: `model/weather.py`

Источник Open-Meteo, без ключа. Каждый ответ сохраняется в `data/weather_cache/<имя>.json`;
при `WEATHER_OFFLINE=1` или при ошибке сети берётся кэш, и это пишется в лог.

```python
WEATHER_COLUMNS = ["ws10", "ws100", "gust10", "dir100", "temp2m", "pressure"]

def get_training_weather(start: str, end: str) -> pd.DataFrame:
    """Архив прогнозов (historical-forecast-api) по WEATHER_POINT, часовой ряд start..end.
    Индекс: time (tz Asia/Almaty). Колонки: WEATHER_COLUMNS. Запросы режутся по годам, кэшируются."""

def get_issued_forecast(issue_date: str, horizon_hours: int = 48) -> pd.DataFrame:
    """Прогноз, известный в день issue_date (previous-runs-api): для часов issue_date+1 берётся
    previous_day1, для issue_date+2 — previous_day2. Ровно horizon_hours строк начиная с
    issue_date+1 00:00. Индекс time, колонки WEATHER_COLUMNS + lead_hours (int, часы от issue_date 00:00)
    + source ("previous_day1"/"previous_day2") + fetched_from ("network"/"cache")."""
```

Переменные Open-Meteo: `wind_speed_10m, wind_speed_100m, wind_gusts_10m, wind_direction_100m,
temperature_2m, surface_pressure`, `wind_speed_unit=ms`, `timezone=Asia/Almaty`.

## Данные и признаки: `model/prepare.py`, `model/features.py`

```python
def load_hourly() -> pd.DataFrame:
    """Читает data/raw/turbine_{1,2}.csv, сводит к часу. Колонки: time, turbine, power (среднее 0..1),
    ws_measured, temp_measured, n_samples (сколько 10-минутных замеров вошло, 0..6)."""

def filter_training(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Убирает часы с n_samples < 4 и вероятные простои (ws_measured >= 6 м/с при power == 0
    не менее 3 часов подряд). Возвращает данные и отчёт: сколько часов убрано по каждой причине."""

FEATURES = ["ws10", "ws100", "gust10", "dir_sin", "dir_cos", "temp2m", "pressure",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "turbine"]

def build_features(weather: pd.DataFrame, turbine: int) -> pd.DataFrame:
    """Из погодного ряда (индекс time, WEATHER_COLUMNS) делает таблицу с колонками FEATURES,
    индекс time сохраняется."""
```

## Обучение и предсказание: `model/train.py`, `model/predict.py`

`python -m model.train` делает всё за один запуск:
1. `load_hourly` → `filter_training`;
2. `get_training_weather(TRAIN_START, "2026-01-31")`;
3. самопроверка выравнивания по времени: корреляция `ws_measured` с `ws100` при сдвигах −2..+2 часа,
   выбирается лучший сдвиг, число пишется в отчёт (ожидаем 0);
4. обучение трёх `HistGradientBoostingRegressor(loss="quantile", quantile=q)` для q в 0.1, 0.5, 0.9
   на всех месяцах, кроме `VALIDATION_MONTH`;
5. проверка на январе 2026 через `get_issued_forecast` для каждой даты выпуска с 31.12.2025 по 30.01.2026
   (то есть ровно так, как агент работает в феврале): MAE, RMSE по p50, доля факта внутри коридора p10–p90;
   рядом две точки отсчёта: persistence («завтра как сегодня») и кривая мощности по ветру из истории;
6. сохраняет `model/artifacts/model_q10.joblib`, `model_q50.joblib`, `model_q90.joblib`,
   `metrics.json`, `validation.csv` (time, turbine, actual, p10, p50, p90, persistence, curve),
   `report.md` (объём данных, что выброшено, сдвиг, метрики против точек отсчёта).

```python
def train(train_start: str | None = None, artifacts_dir: str | None = None) -> dict
    """Полный цикл обучения, возвращает метрики. CLI зовёт с умолчаниями (TRAIN_START, ARTIFACTS).
    Тест зовёт train(train_start="2025-11-02", artifacts_dir=<временная папка>), не трогая боевые артефакты."""
def load_models(artifacts_dir: str | None = None) -> dict   # {"q10","q50","q90"}; кэшируется на процесс;
                                                            # если артефактов нет — PowerCurveModel
def predict(features: pd.DataFrame, turbine: int, models: dict | None = None) -> pd.DataFrame
    """Индекс time; колонки: turbine, p10, p50, p90 в [0,1], p10 <= p50 <= p90 (после сортировки)."""
```

`PowerCurveModel` — кривая мощности по бинам ветра из истории; она же точка отсчёта и запасной
предсказатель, если артефактов нет.

## Агент: `agent/`

```python
# agent/tools.py — каждый инструмент чистая функция с логом входа/выхода
fetch_weather(issue_date) -> pd.DataFrame          # обёртка над get_issued_forecast
prepare(weather) -> dict[int, pd.DataFrame]          # признаки по турбинам
run_model(features_by_turbine) -> pd.DataFrame       # объединённый прогноз обеих турбин
save_forecast(issue_date, forecast, weather) -> str  # путь к output/forecasts/forecast_<issue_date>.csv
analyze(issue_date, forecast, previous_forecast, actuals) -> dict
    # totals по турбинам и суткам; delta_vs_previous по пересекающимся часам; low_confidence_hours
    # (|p50 − p50_prev| > 0.15 или ширина коридора p90−p10 > 0.7, это верхняя четверть часов при медиане 0.50); extreme_wind_hours (ws100 > 25 м/с);
    # error_yesterday (MAE по факту, если факт есть, иначе None)
write_journal(issue_date, analysis, note) -> None    # добавляет раздел в output/journal.md

# agent/run.py
def run_day(issue_date: str) -> RunResult
    """RunResult: issue_date, steps (list of {name, started, finished, summary}), forecast_path,
    analysis (dict), note (str). Последовательность: fetch_weather → prepare → run_model →
    save_forecast → analyze → note → write_journal. Записывает output/runs/<issue_date>.json."""
def run_period(start: str, end: str) -> list[RunResult]
# CLI: python -m agent.run --date 2026-02-05 ; python -m agent.run --from 2026-01-31 --to 2026-02-28

# agent/analyst.py — только при OPENAI_API_KEY
def daily_note(analysis: dict) -> str     # без ключа: шаблонная сводка на русском из чисел analysis
def ask(question: str, history: list) -> str
    # инструменты для модели: get_forecast(issue_date), get_weather(issue_date), compare_runs(d1, d2),
    # get_validation(); ответ по-русски, только по данным инструментов
```

Формат `output/forecasts/forecast_<issue_date>.csv`, колонки строго:
`issue_date, target_time, lead_hours, turbine, ws100_forecast, temp_forecast, p10, p50, p90, confidence, note`
(`confidence` ∈ {"ok", "low"}; 96 строк: 48 часов × 2 турбины; `target_time` в ISO с смещением).

## Веб: `web/app.py`, `web/static/index.html`

`python -m web.app` поднимает uvicorn на `0.0.0.0:8000`. При старте: если артефактов нет, запускает
обучение (из кэша погоды) и пишет об этом в лог.

```
GET  /                         страница
GET  /api/status               {model_ready, weather_cache_ok, llm_enabled, offline}
POST /api/run?issue_date=...   RunResult в JSON + rows прогноза
POST /api/run-period?from=&to= список RunResult (синхронно; на кэше это секунды)
GET  /api/forecast/{issue_date}         JSON строк прогноза
GET  /api/forecast/{issue_date}.csv     файл
GET  /api/journal              текст output/journal.md
GET  /api/validation           metrics.json + строки validation.csv
POST /api/ask                  {question, history} → {answer}; 503 с текстом, если ключа нет
```

Страница: выбор даты (31.01–28.02.2026), кнопка «Сформировать прогноз», журнал шагов, график
48 часов с коридором p10–p90 и линией p50 по каждой турбине (SVG без внешних библиотек), таблица,
ссылка на CSV, кнопка «Прогнать весь февраль», панель «Проверка на январе», окно вопросов агенту
(скрыто, если `llm_enabled=false`). Ошибки показываются текстом, страница не падает на кривом вводе
(дата вне периода, пустая дата).

## Тесты и примеры

`tests/test_scenario.py`: обучение на срезе данных (последние 90 дней), `run_day("2026-02-05")`
при `WEATHER_OFFLINE=1`, проверка: файл есть, 96 строк, значения в [0,1], p10 ≤ p50 ≤ p90,
в конце `print("PASS")`. `examples/`: один готовый `forecast_2026-02-05.csv`, фрагмент журнала,
`validation_summary.md`.

## Упаковка

`requirements.txt` с точными версиями; `Dockerfile` от `python:3.12-slim`; `docker-compose.yml`
с сервисом `app`, порт 8000, `.env` подхватывается через `env_file`, `output/` смонтирован наружу.
