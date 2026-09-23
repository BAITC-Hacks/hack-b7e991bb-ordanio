# Общие константы проекта: координаты, часовой пояс, пути к данным и артефактам, границы периодов.
# Все пути относительные от корня репозитория.

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
