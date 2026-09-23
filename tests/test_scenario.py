# Проверочный сценарий: обучение на срезе (последние 90 дней) во временную папку, прогон run_day("2026-02-05")
# без сети (WEATHER_OFFLINE=1), проверка формы CSV (96 строк, [0,1], p10 <= p50 <= p90) и кривого ввода; в конце PASS.

import importlib
import inspect
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["WEATHER_OFFLINE"] = "1"          # до импортов проекта: погода только из data/weather_cache
os.chdir(ROOT)                                # пути проекта относительные от корня репозитория
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

ISSUE_DATE = "2026-02-05"
TZ = "Asia/Almaty"                           # как common.config.TZ
SLICE_START = "2025-11-02"                   # 90 дней до 2026-01-31
EXPECTED_COLUMNS = ["issue_date", "target_time", "lead_hours", "issue_timestamp", "weather_lead_hours",
                    "weather_run_time", "turbine", "ws100_forecast", "temp_forecast", "p10", "p50", "p90",
                    "confidence", "note"]
NUMERIC_COLUMNS = ["lead_hours", "weather_lead_hours", "turbine", "ws100_forecast", "temp_forecast",
                   "p10", "p50", "p90"]
EXPECTED_STEPS = ["fetch_weather", "prepare", "run_model", "save_forecast", "analyze", "write_journal"]
ARTIFACT_FILES = ["model_q10.joblib", "model_q50.joblib", "model_q90.joblib", "metrics.json"]
SKIP_TRAIN = "model.train пока не принимает срез и папку артефактов"
CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _snapshot(folder: Path) -> dict:
    """Имена и время изменения файлов папки: так проверяем, что боевые артефакты не тронуты."""
    if not folder.exists():
        return {}
    return {p.name: p.stat().st_mtime_ns for p in folder.iterdir() if p.is_file()}


@pytest.fixture(scope="session")
def slice_training(tmp_path_factory):
    """Обучает модель на срезе во временную папку. Возвращает (папка, метрики) или (None, причина пропуска).
    Полное обучение на трёх годах здесь не запускается никогда."""
    try:
        train_module = importlib.import_module("model.train")
    except ModuleNotFoundError as exc:
        if exc.name in ("model.train", "model.features"):
            return None, f"{SKIP_TRAIN} (нет модуля {exc.name})"
        raise
    fn = getattr(train_module, "train", None)
    if fn is None:
        return None, f"{SKIP_TRAIN} (в model.train нет функции train)"
    params = inspect.signature(fn).parameters
    if "train_start" not in params or "artifacts_dir" not in params:
        return None, f"{SKIP_TRAIN} (сигнатура train{inspect.signature(fn)})"

    from common.config import ARTIFACTS
    before = _snapshot(ROOT / ARTIFACTS)
    out_dir = tmp_path_factory.mktemp("artifacts_slice")
    metrics = fn(train_start=SLICE_START, artifacts_dir=str(out_dir))
    assert _snapshot(ROOT / ARTIFACTS) == before, "обучение на срезе изменило боевые model/artifacts"
    return out_dir, metrics


@pytest.fixture(scope="session")
def agent_output(tmp_path_factory):
    """Временная папка вместо output/: туда агент пишет прогноз, итог дня и журнал во время теста.
    Вчерашний прогноз (если есть в боевой папке) копируется, чтобы сравнение с ним тоже работало."""
    import shutil
    from agent import tools

    out = tmp_path_factory.mktemp("output")
    paths = {"FORECASTS_DIR": out / "forecasts", "RUNS_DIR": out / "runs", "JOURNAL_PATH": out / "journal.md"}
    paths["FORECASTS_DIR"].mkdir()
    paths["RUNS_DIR"].mkdir()
    prev_date = (pd.Timestamp(ISSUE_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prev = Path(tools.FORECASTS_DIR) / f"forecast_{prev_date}.csv"
    if prev.exists():
        shutil.copy2(prev, paths["FORECASTS_DIR"] / prev.name)
    with pytest.MonkeyPatch.context() as mp:
        for name, path in paths.items():
            mp.setattr(tools, name, str(path))
        yield paths


@pytest.fixture(scope="session")
def scenario(slice_training, agent_output):
    """Один прогон run_day на дату из тестового периода, вывод во временную папку (output/ не трогается).
    Если срез обучен, агент работает на его моделях (подставляются в кэш model.predict на время сессии);
    иначе как в бою: артефакты или PowerCurveModel."""
    from model import predict as predict_module
    from agent.run import run_day

    out_dir, _ = slice_training
    slice_models = None
    if out_dir is not None:
        slice_models = predict_module.load_models(str(out_dir))
        predict_module._MODELS_CACHE = slice_models
    try:
        yield run_day(ISSUE_DATE)
    finally:
        # Кэш процесса возвращается на боевые model/artifacts: временные модели в нём не остаются.
        predict_module.load_models(refresh=True)
        if slice_models is not None:
            assert predict_module._MODELS_CACHE is not slice_models, "кэш моделей остался на срезе"


def test_train_on_slice(slice_training):
    out_dir, info = slice_training
    if out_dir is None:
        pytest.skip(info)
    for name in ARTIFACT_FILES:
        assert (out_dir / name).exists(), f"после обучения на срезе нет {name}"
    assert isinstance(info, dict), "train должен возвращать словарь метрик"


def test_run_day_forecast(scenario, agent_output):
    result = scenario
    assert result.issue_date == ISSUE_DATE
    path = Path(result.forecast_path)
    assert path.exists(), f"файла прогноза нет: {path}"
    assert path.resolve().parent == agent_output["FORECASTS_DIR"].resolve(), f"прогноз записан не во временную папку: {path}"
    assert path.name == f"forecast_{ISSUE_DATE}.csv"

    df = pd.read_csv(path, keep_default_na=False)
    assert len(df) == 96, f"ожидалось 96 строк, получено {len(df)}"
    assert list(df.columns) == EXPECTED_COLUMNS, f"колонки не по контракту: {list(df.columns)}"
    assert (df["issue_date"].astype(str) == ISSUE_DATE).all()

    # Числовые колонки: ни пропусков, ни нечисловых значений.
    for col in NUMERIC_COLUMNS:
        bad = pd.to_numeric(df[col], errors="coerce").isna()
        assert not bad.any(), f"{col}: пропуски или нечисловые значения в строках {list(df.index[bad])[:5]}"

    assert df["turbine"].value_counts().to_dict() == {1: 48, 2: 48}
    assert not df.duplicated(["turbine", "target_time"]).any(), "пары (turbine, target_time) повторяются"
    assert len(df[["turbine", "target_time"]].drop_duplicates()) == 96, "уникальных пар (turbine, target_time) не 96"

    target = pd.to_datetime(df["target_time"], utc=True).dt.tz_convert(TZ)
    issue_ts = pd.to_datetime(df["issue_timestamp"], utc=True)
    run_time = pd.to_datetime(df["weather_run_time"], utc=True)
    w_lead = pd.to_numeric(df["weather_lead_hours"], errors="coerce")
    for col, parsed in (("target_time", target), ("issue_timestamp", issue_ts), ("weather_run_time", run_time)):
        assert parsed.notna().all(), f"{col}: есть нечитаемые даты"

    day1 = pd.Timestamp(ISSUE_DATE, tz=TZ) + pd.Timedelta(days=1)
    expected_hours = pd.date_range(day1, periods=48, freq="h", tz=TZ)
    for turbine in (1, 2):
        mask = df["turbine"].astype(int) == turbine
        got = pd.DatetimeIndex(target[mask].sort_values().values).tz_localize("UTC").tz_convert(TZ)
        assert got.equals(expected_hours), (f"турбина {turbine}: target_time не ряд {expected_hours[0]} … "
                                            f"{expected_hours[-1]} по часу")
        lead = sorted(pd.to_numeric(df.loc[mask, "lead_hours"]).astype(int))
        assert lead == list(range(24, 72)), f"турбина {turbine}: lead_hours не 24..71"

    expected_issue = pd.Timestamp(f"{ISSUE_DATE} 23:59", tz=TZ)
    assert (issue_ts == expected_issue).all(), f"issue_timestamp не {expected_issue.isoformat()}: {set(df['issue_timestamp'])}"

    is_day1 = target < day1 + pd.Timedelta(days=1)
    expected_w_lead = pd.Series(48, index=df.index).where(~is_day1, 24)
    wrong = w_lead != expected_w_lead
    assert not wrong.any(), f"weather_lead_hours не 24 для D+1 и 48 для D+2: строки {list(df.index[wrong])[:5]}"
    expected_run = target.dt.tz_convert("UTC") - pd.to_timedelta(w_lead, unit="h")
    wrong = run_time != expected_run
    assert not wrong.any(), (f"weather_run_time != target_time − weather_lead_hours: "
                             f"{df.loc[wrong, ['target_time', 'weather_lead_hours', 'weather_run_time']].head(3).to_dict('records')}")
    late = run_time > issue_ts
    assert not late.any(), (f"weather_run_time позже issue_timestamp: "
                            f"{df.loc[late, ['target_time', 'weather_run_time', 'issue_timestamp']].head(3).to_dict('records')}")

    q = df[["p10", "p50", "p90"]].astype(float)
    assert q.notna().all().all(), "в p10/p50/p90 есть пропуски"
    assert ((q >= 0) & (q <= 1)).all().all(), "значения p10/p50/p90 вне [0, 1]"
    eps = 1e-9
    assert (q["p10"] <= q["p50"] + eps).all(), "p10 > p50"
    assert (q["p50"] <= q["p90"] + eps).all(), "p50 > p90"
    assert set(df["confidence"]) <= {"ok", "low"}, f"confidence вне ok/low: {set(df['confidence'])}"

    names = [s["name"] for s in result.steps]
    assert len(result.steps) >= 6, f"шагов меньше шести: {names}"
    positions = [names.index(n) if n in names else -1 for n in EXPECTED_STEPS]
    assert -1 not in positions, f"не хватает шагов: {names}"
    assert positions == sorted(positions), f"шаги не по порядку: {names}"
    for s in result.steps:
        assert {"name", "started", "finished", "summary"} <= set(s), f"у шага {s.get('name')} не все поля"
    assert (agent_output["RUNS_DIR"] / f"{ISSUE_DATE}.json").exists(), "итог дня не записан во временную папку"
    assert agent_output["JOURNAL_PATH"].exists(), "журнал не записан во временную папку"


@pytest.mark.parametrize("bad", ["2026-03-05", "abc", ""])
def test_bad_input(bad):
    from agent.run import run_day
    with pytest.raises(ValueError) as err:
        run_day(bad)
    assert CYRILLIC.search(str(err.value)), f"сообщение не по-русски: {err.value}"


def test_zz_pass(request):
    """Последний тест: печатает PASS, только если ни одна проверка выше не упала."""
    assert request.session.testsfailed == 0, "есть упавшие проверки, PASS не печатается"
    print("PASS")
