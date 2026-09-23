# Протокол чистого запуска

Выполнено 23.09.2026 в 15:47 и 16:00 на macOS 15 (Apple Silicon), Docker 29.4, Python 3.12.14.
Репозиторий скачан заново с GitHub во временную папку, ключ OpenAI не задавался, `.env` отсутствовал.
Ниже сокращённый протокол проверок: команды, основные результаты и пояснения автора, это не полный
терминальный лог. Исторические результаты ранней версии (режим по умолчанию 24/48) отделены от проверок
финальных коммитов; копируемые команды для повторения приведены в README, раздел «Как проверить».
Общий md5 CSV считался как md5 конкатенации содержимого файлов `output/forecasts/*.csv` в порядке
сортировки имён; он служит внутренним сравнением двух прогонов, а не самостоятельным критерием.

## Коммит

```
git clone https://github.com/BAITC-Hacks/hack-b7e991bb-ordanio.git repo
git log --oneline -1
8e30e31 Примеры из финального прогона; README: возможности страницы и проверенное окружение
```

После перехода основного режима на 48/72 функциональные проверки повторены на `f2e91a4` и `169e02c`;
актуальные метрики и состав выходов приведены в разделе «Финальный прогон» ниже. Более ранние результаты
в этом файле относятся к прежнему режиму по умолчанию (24/48) и оставлены как история.

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
58   (две строки на выпуск; формулировка в текущей версии: «По номинальному упреждению все значения
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

## Финальный прогон на замороженном коммите (16:39–16:41)

```
git log --oneline -1
f2e91a4 Основной режим погоды 48/72 (строгий): умолчание, артефакты и проверки января и декабря, ...
Python 3.12.14

WEATHER_OFFLINE=1 .venv/bin/python -m pytest -q -s tests
6 passed in 40.38s   (PASS печатается; output/ после теста пустой)

WEATHER_OFFLINE=1 .venv/bin/python -m agent.run --from 2026-01-31 --to 2026-02-28
Готово: 29 дней, файлы в output/forecasts, журнал output/journal.md
файлов 29, строк 2784, в феврале 2640, уникальных пар час–турбина в феврале 1344, за 1–2 марта 144,
упреждения {48, 72}, weather_run_time <= issue_timestamp во всех строках: True
md5 всех CSV: 12d95fdf2bc0a5dbdfcd69813116b26e (совпадает с прогоном прораба в 16:33)
журнал: «Режим погоды: строгий, упреждение 48/72 ч.» в каждом разделе

Два последовательных выпуска 04.02 и 05.02: общих записей 48, упреждение погоды вчера 72 ч,
сегодня 48 ч, изменение p50 больше 0.15 в 8 записях.

WEATHER_STRICT=0 WEATHER_OFFLINE=1 .venv/bin/python -m agent.run --date 2026-02-05
упреждения в CSV: {24, 48}
(сравнительный запуск заменил файл 05.02 в общей папке output/ этой временной копии; основной набор
после него не восстанавливался, дальнейшие шаги протокола от output/ не зависят)

WEATHER_OFFLINE=1 .venv/bin/python -m model.train --validation-month 2025-12 --artifacts-dir <временная>
декабрь: MAE 0.2061, кривая 0.2146, покрытие 0.765

mv model/artifacts <в сторону>; WEATHER_OFFLINE=1 .venv/bin/python -m model.train
январь (48/72): MAE 0.1603, кривая 0.1765, покрытие 0.826, weather_strict: true
md5 model_q50.joblib: новый 2060bcca8d5b0e26ab36a7a7edaa8386 = в репозитории

docker compose up --build -d → /, /api/forecasts.csv, /api/journal.md, /api/validation: 200;
POST /api/run?issue_date=abc: 400
```
