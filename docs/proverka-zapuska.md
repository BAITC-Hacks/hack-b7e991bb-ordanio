# Протокол чистого запуска

Выполнено 23.09.2026 в 15:47 и 16:00 на macOS 15 (Apple Silicon), Docker 29.4, Python 3.12.14.
Репозиторий скачан заново с GitHub во временную папку, ключ OpenAI не задавался,
`.env` отсутствовал. Ниже команды и вывод дословно (сокращены только пути временных папок).

## Коммит

```
git clone https://github.com/BAITC-Hacks/hack-b7e991bb-ordanio.git repo
git log --oneline -1
8e30e31 Примеры из финального прогона; README: возможности страницы и проверенное окружение
```

Повторено на `a8dc2e3` и на финальном коммите (тесты, весь период в основном режиме 48/72: 2 784 строки,
2 640 внутри февраля, 1 344 уникальные пары, 144 строки за 1–2 марта; строгий и сравнительный режимы;
воспроизведение декабря; Docker, все маршруты) с тем же результатом; вывод финального прогона добавлен ниже.

## Окружение и тест

```
uv venv --seed --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python --version
Python 3.12.14

WEATHER_OFFLINE=1 .venv/bin/python -m pytest -q -s tests
.....PASS
.
6 passed in 39.98s

ls output
(пусто: тест не пишет в боевую папку)
```

## Один день агента

```
WEATHER_OFFLINE=1 .venv/bin/python -m agent.run --date 2026-02-05
Готово: output/forecasts/forecast_2026-02-05.csv, журнал output/journal.md
```

Файл на 96 строк. Его md5 отличается от `examples/forecast_2026-02-05.csv`, потому что пример взят
из прогона периода, где у 5 февраля есть предыдущий выпуск и заполнено сравнение; значения
p10/p50/p90 совпадают, отличается колонка `confidence`.

## Весь период

```
WEATHER_OFFLINE=1 .venv/bin/python -m agent.run --from 2026-01-31 --to 2026-02-28
Готово: 29 дней, файлы в output/forecasts, журнал output/journal.md

ls output/forecasts | wc -l
29
grep -c "не позже момента выпуска" output/journal.md
29   (по одной строке на выпуск; формулировка в текущей версии: «По номинальному упреждению все значения
     погоды рассчитаны не позже момента выпуска …»)
```

## Docker

```
docker compose up --build -d
 Container repo-app-1 Started

curl -s http://localhost:8000/api/status
{"model_ready":true,"weather_cache_ok":true,"llm_enabled":false,"offline":false, ...}

curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/            → 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/forecasts.csv → 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/journal.md   → 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/validation   → 200
curl -s -X POST "http://localhost:8000/api/run?issue_date=2026-02-10"
{"issue_date":"2026-02-10","steps":[{"name":"fetch_weather", ...

curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:8000/api/run?issue_date=abc" → 400
```

## Обучение с нуля из кэша погоды

```
mv model/artifacts /tmp/art && WEATHER_OFFLINE=1 .venv/bin/python -m model.train
15:47:37 INFO train: Готово за 12.4 с. Файлы: metrics.json, model_q10.joblib, model_q50.joblib,
model_q90.joblib, power_curve.joblib, report.md, validation.csv

metrics.json: MAE 0.14404026601208036, покрытие 0.8534660260809883
md5 model_q50.joblib: новый 2060bcca8d5b0e26ab36a7a7edaa8386, в репозитории 2060bcca8d5b0e26ab36a7a7edaa8386
```

Совпадение md5 проверено на одной машине; на другой ОС и процессоре числа могут отличаться
в последних знаках, метрики при этом должны совпасть до третьего знака.

## Кривой ввод в командной строке

```
.venv/bin/python -m agent.run --date 2026-03-05
Ошибка входных данных: Дата 2026-03-05 вне тестового периода: допустимы даты с 2026-01-31 по 2026-02-28.
(код выхода 2)

.venv/bin/python -m agent.run --date abc
Ошибка входных данных: Дата «abc» не в формате ГГГГ-ММ-ДД, например 2026-02-05.
(код выхода 2)
```
