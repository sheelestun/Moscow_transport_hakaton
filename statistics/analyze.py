"""Reproducible, read-only dataset audit. Run from any working directory.

Outputs contain aggregate statistics, never raw telemetry or inferred validate labels.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "test", "validate")
TIME_COLUMNS = {"event_time", "gps_time", "receive_time", "time_begin", "time_fact_begin", "T", "target_time_begin"}
NAV_COLUMNS = ["gps_time", "lon", "lat", "alt", "speed", "heading"]
REQUIRED = {
    "traffic": {"packet_id", "tr_id", "unit_id", "event_time", "receive_time", "location_valid", *NAV_COLUMNS},
    "schedule": {"tt_action_item_id", "tr_id", "time_begin", "geom"},
    "points": {"sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s"},
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(split: str) -> dict[str, str]:
    return {
        "traffic": f"{split}/traffic.csv",
        "schedule": f"{split}/{'schedule_plan.csv' if split == 'validate' else 'schedule.csv'}",
        "points": "validate/points.csv" if split == "validate" else f"labels/labels_{split}.csv",
    }


def read_source(path: Path, kind: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep packet identifiers textual; count invalid timestamps separately from NULL."""
    df = pd.read_csv(path, dtype={"packet_id": "string", "sample_id": "string"}, low_memory=False)
    missing = REQUIRED[kind] - set(df)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    invalid = {}
    for name in TIME_COLUMNS.intersection(df):
        original = df[name]
        parsed = pd.to_datetime(original, format="mixed", errors="coerce")
        invalid[name] = int((original.notna() & parsed.isna()).sum())
        if invalid[name]:
            raise ValueError(f"{path.name}: {invalid[name]} malformed timestamps in {name}")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype) or not pd.api.types.is_datetime64_dtype(parsed):
            raise ValueError(f"{path.name}: expected naive homogeneous timestamps; confirm timezone contract")
        df[name] = parsed
    if kind == "traffic":
        if df.location_valid.isna().any() or not df.location_valid.isin([True, False]).all():
            raise ValueError("location_valid must contain booleans without NULL")
        df["location_valid"] = df.location_valid.astype(bool)
        # A diagnostic heuristic, not an authoritative origin label.
        df["cohort"] = np.where(df.packet_id.str.fullmatch(r"-?\d+", na=False), "real", "synthetic_candidate")
    return df, invalid


def point_coverage(points: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Per-point coverage using event_time <= T and the open-left (T-900s,T] window.

    Nanosecond int64 comparisons avoid timestamp precision loss. Invalid/missing
    coordinates are excluded even if the other navigation fields are populated.
    """
    result = points.copy()
    valid = traffic.loc[traffic.location_valid & traffic.lon.notna() & traffic.lat.notna() & traffic.event_time.notna()]
    groups = {
        vehicle: np.sort(group.event_time.to_numpy(dtype="datetime64[ns]").view("int64"))
        for vehicle, group in valid.groupby("tr_id")
    }
    ages, counts = [], []
    for row in points.itertuples(index=False):
        values = groups.get(row.tr_id, np.array([], dtype=np.int64))
        if pd.isna(row.T):
            ages.append(np.nan)
            counts.append(0)
            continue
        now = row.T.value
        right = int(np.searchsorted(values, now, side="right"))
        left = int(np.searchsorted(values, now - 900_000_000_000, side="right"))
        ages.append((now - values[right - 1]) / 1e9 if right else np.nan)
        counts.append(right - left)
    result["valid_ping_age_s"] = ages
    result["valid_pings_15m"] = counts
    return result


def numeric_description(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    result = {"n": int(v.size), "missing_or_nonfinite": int(len(values) - v.size)}
    for name, q in [("min", 0), ("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("p99", .99), ("max", 1)]:
        result[name] = float(v.quantile(q)) if len(v) else None
    result["mean"] = float(v.mean()) if len(v) else None
    return result


def quality_flags(t: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic flags only; 120 km/h is not an agreed cleaning threshold."""
    return pd.DataFrame({
        "invalid_location": ~t.location_valid,
        "missing_navigation": t[NAV_COLUMNS].isna().any(axis=1),
        "speed_gt_120": t.speed.gt(120),
        "receive_before_event": t.receive_time.lt(t.event_time),
        "gps_after_event": t.gps_time.gt(t.event_time),
    })


def audit(dataset: Path) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    frames, manifests = {}, []
    overview, nulls, numeric, quality, baseline, hourly, intersections, checks = [], [], [], [], [], [], [], []

    def check(split: str, name: str, failures: int, total: int, severity: str = "error"):
        checks.append(dict(split=split, check=name, failures=int(failures), total=int(total), severity=severity))

    for split in SPLITS:
        group = {}
        for kind, relative in source_paths(split).items():
            path = dataset / relative
            df, invalid = read_source(path, kind)
            source_columns = [c for c in df if c != "cohort"]
            manifests.append(dict(path=relative, sha256=file_hash(path), bytes=path.stat().st_size, rows=len(df), columns=source_columns))
            for col in source_columns:
                n = int(df[col].isna().sum())
                nulls.append(dict(split=split, source=kind, column=col, rows=len(df), missing=n, missing_pct=100*n/len(df), dtype=str(df[col].dtype)))
            group[kind] = df
        t, s, p = group["traffic"], group["schedule"], group["points"]
        synthetic_ids = set(t.loc[t.cohort == "synthetic_candidate", "tr_id"])
        p["cohort"] = np.where(p.tr_id.isin(synthetic_ids), "synthetic_candidate", "real")
        p = point_coverage(p, t)
        group["points"] = p
        frames[split] = group
        for kind, df, key in [("traffic", t, "packet_id"), ("schedule", s, "tt_action_item_id"), ("points", p, "sample_id")]:
            check(split, f"{kind}_duplicate_{key}", df[key].duplicated().sum(), len(df))
            check(split, f"{kind}_missing_{key}", df[key].isna().sum(), len(df))
        for kind, df, col in [("traffic", t, "event_time"), ("schedule", s, "time_begin"), ("points", p, "T")]:
            check(split, f"{kind}_missing_time", df[col].isna().sum(), len(df))

        # Only join each split to its OWN schedule. Never infer validate truth.
        joined = p.merge(s, left_on="target_stop_id", right_on="tt_action_item_id", how="left", suffixes=("", "_schedule"), indicator=True, validate="many_to_one")
        check(split, "target_schedule_missing", (joined["_merge"] != "both").sum(), len(p))
        check(split, "target_vehicle_mismatch", (joined.tr_id != joined.tr_id_schedule).sum(), len(p))
        check(split, "target_plan_time_mismatch", (joined.target_time_begin != joined.time_begin).sum(), len(p))
        horizon = (p.target_time_begin-p["T"]).dt.total_seconds()
        check(split, "horizon_outside_open_600_closed_900", (~horizon.gt(600) | ~horizon.le(900)).sum(), len(p))
        if split != "validate":
            expected = (joined.time_fact_begin - joined.time_begin).dt.total_seconds()
            check(split, "target_delay_mismatch_or_missing", (expected.isna() | joined.target_delay_s.isna() | (expected-joined.target_delay_s).abs().gt(1e-6)).sum(), len(p))
        else:
            check(split, "validate_contains_target_column", int("target_delay_s" in p or "time_fact_begin" in s), 1)

        flag = quality_flags(t)
        patterns = flag.value_counts(sort=True).rename("rows").reset_index()
        patterns.insert(0, "split", split)
        intersections.extend(patterns.to_dict("records"))
        receive_lag = (t.receive_time-t.event_time).dt.total_seconds()
        gps_lag = (t.event_time-t.gps_time).dt.total_seconds()
        gap = t.sort_values(["tr_id", "event_time"]).groupby("tr_id").event_time.diff().dt.total_seconds()
        metrics = {
            **{c: int(flag[c].sum()) for c in flag},
            "invalid_with_coordinates": int((~t.location_valid & t.lon.notna() & t.lat.notna()).sum()),
            "valid_missing_coordinates": int((t.location_valid & (t.lon.isna() | t.lat.isna())).sum()),
            "speed_negative": int(t.speed.lt(0).sum()),
            "coordinates_outside_global_bounds": int((t.lon.abs().gt(180) | t.lat.abs().gt(90)).sum()),
            "heading_outside_0_360": int((t.heading.lt(0) | t.heading.ge(360)).sum()),
            "receive_lag_gt_60s": int(receive_lag.gt(60).sum()),
            "extra_vehicle_time_duplicates": int(t.duplicated(["tr_id", "event_time"]).sum()),
            "positive_gap_gt_300s": int(gap.gt(300).sum()),
        }
        for metric, count in metrics.items():
            quality.append(dict(split=split, metric=metric, count=count, denominator=len(t), pct=100*count/len(t)))
        for cohort, ct in [("all", t), *t.groupby("cohort", sort=True)]:
            cp = p if cohort == "all" else p.loc[p.cohort == cohort]
            overview.append(dict(split=split, cohort=cohort, traffic_rows=len(ct), vehicles=ct.tr_id.nunique(), schedule_rows=len(s) if cohort == "all" else int(s.tr_id.isin(set(ct.tr_id)).sum()), points=len(cp), point_vehicles=cp.tr_id.nunique(), event_min=str(ct.event_time.min()), event_max=str(ct.event_time.max()), points_no_valid_history=int(cp.valid_ping_age_s.isna().sum()), points_no_valid_ping_15m=int(cp.valid_pings_15m.eq(0).sum())))
            measures = {
                "speed_kmh": ct.speed,
                "receive_minus_event_s": (ct.receive_time-ct.event_time).dt.total_seconds(),
                "event_minus_gps_s": (ct.event_time-ct.gps_time).dt.total_seconds(),
                "valid_ping_age_s": cp.valid_ping_age_s,
                "valid_pings_15m": cp.valid_pings_15m,
            }
            if cohort == "all":
                measures.update(positive_event_gap_s=gap[gap > 0], horizon_s=horizon)
            if "target_delay_s" in cp:
                measures.update(target_delay_s=cp.target_delay_s, baseline_abs_error_s=(cp.target_delay_s-cp.cur_dev_s).abs())
            for metric, values in measures.items():
                numeric.append(dict(split=split, cohort=cohort, metric=metric, **numeric_description(values)))
        ht = t.assign(hour=t.event_time.dt.hour).groupby("hour").agg(rows=("tr_id", "size"), vehicles=("tr_id", "nunique"), valid_location=("location_valid", "sum")).reset_index()
        ht.insert(0, "split", split)
        hourly.extend(ht.to_dict("records"))
        if split != "validate":
            for dimension, groups in [("cohort", [("all", p), *p.groupby("cohort")]), ("class", p.groupby("target_class")), ("hour", p.groupby(p["T"].dt.hour))]:
                for value, subset in groups:
                    baseline.append(dict(split=split, dimension=dimension, group=str(value), points=len(subset), mae_cur_dev_s=float((subset.target_delay_s-subset.cur_dev_s).abs().mean()), mae_zero_s=float(subset.target_delay_s.abs().mean())))

    overlaps = []
    for a, b in itertools.combinations(SPLITS, 2):
        for kind, col in [("traffic", "packet_id"), ("schedule", "tt_action_item_id"), ("points", "sample_id"), ("points", "target_stop_id")]:
            x, y = set(frames[a][kind][col].dropna()), set(frames[b][kind][col].dropna())
            overlaps.append(dict(split_a=a, split_b=b, entity=col, unique_a=len(x), unique_b=len(y), intersection=len(x & y), union=len(x | y)))

    template_path = dataset / "sample_submission.csv"
    template = pd.read_csv(template_path, sep=";", dtype={"sample_id": "string"})
    manifests.append(dict(path="sample_submission.csv", sha256=file_hash(template_path), bytes=template_path.stat().st_size, rows=len(template), columns=template.columns.tolist()))
    wanted = set(frames["validate"]["points"].sample_id)
    check("validate", "template_duplicate_ids", template.sample_id.duplicated().sum(), len(template))
    check("validate", "template_id_set_difference", len(wanted ^ set(template.sample_id)), len(wanted))
    check("validate", "template_nonfinite_prediction", (~np.isfinite(template.prediction)).sum(), len(template))
    tables = {"overview": pd.DataFrame(overview), "missing_values": pd.DataFrame(nulls), "numeric_summary": pd.DataFrame(numeric), "quality_flags": pd.DataFrame(quality), "baseline_metrics": pd.DataFrame(baseline), "hourly_traffic": pd.DataFrame(hourly), "quality_intersections": pd.DataFrame(intersections), "split_overlap": pd.DataFrame(overlaps), "checks": pd.DataFrame(checks)}
    manifest = dict(schema_version=1, seed=42, input_files=manifests, python=platform.python_version(), libraries={name: importlib.metadata.version(name) for name in ["pandas", "numpy", "matplotlib", "scipy"]}, assumptions={"timezone": "naive source timestamps; no timezone conversion", "synthetic_cohort": "non-numeric packet_id heuristic, propagated via tr_id", "speed_flag_kmh": 120, "coverage_window": "(T-900s,T] using event_time and valid coordinates", "validate_truth": "not loaded or reconstructed", "output_scope": "aggregates only; plotted rain points are sampled distribution values, no identifiers"})
    return frames, tables, manifest


def conclusions(tables: dict[str, pd.DataFrame]) -> list[str]:
    overview = tables["overview"]
    q = tables["quality_flags"]
    def count(split, metric):
        return int(q.loc[(q.split == split) & (q.metric == metric), "count"].iloc[0])
    train = overview.loc[(overview.split == "train") & (overview.cohort == "all")].iloc[0]
    test = overview.loc[(overview.split == "test") & (overview.cohort == "all")].iloc[0]
    validate = overview.loc[(overview.split == "validate") & (overview.cohort == "all")].iloc[0]
    synthetic = overview.loc[(overview.split == "train") & (overview.cohort == "synthetic_candidate")]
    lines = [
        f"Объём: train {train.traffic_rows:,} событий / {train.points:,} прогнозных точек; test {test.traffic_rows:,} / {test.points}; validate {validate.traffic_rows:,} / {validate.points}.",
        f"Невалидные координаты: train {count('train','invalid_location'):,} ({100*count('train','invalid_location')/train.traffic_rows:.2f}%), test {count('test','invalid_location'):,} ({100*count('test','invalid_location')/test.traffic_rows:.2f}%). В train {count('train','invalid_with_coordinates'):,} таких строк всё же имеют lon/lat: проверки на NULL недостаточно.",
        f"Без валидного GPS в (T−15 минут,T]: train {train.points_no_valid_ping_15m}, test {test.points_no_valid_ping_15m}, validate {validate.points_no_valid_ping_15m}. Без всей предшествующей валидной GPS-истории: train {train.points_no_valid_history}.",
        f"receive_time раньше event_time: train {count('train','receive_before_event'):,}, test {count('test','receive_before_event'):,}. Доступность сообщения и время события требуют разных полей и правил.",
        f"Скорость выше диагностических 120 км/ч: train {count('train','speed_gt_120')}, test {count('test','speed_gt_120')}. Это флаг для изучения, не автоматическое правило удаления.",
        f"Лишние повторения пары (tr_id,event_time): train {count('train','extra_vehicle_time_duplicates'):,}, test {count('test','extra_vehicle_time_duplicates'):,}. Разные packet_id нельзя удалять только по совпадению этой пары.",
    ]
    if len(synthetic):
        row = synthetic.iloc[0]
        lines.append(f"Предположительно синтетическая когорта train: {row.traffic_rows:,} событий, {row.vehicles} ТС и {row.points:,} точек. Определена по нечисловому packet_id и связи tr_id. Наличие синтетики подтверждено README, точный признак происхождения не дан.")
    for row in tables["baseline_metrics"].query("dimension == 'cohort' and group == 'all'").itertuples():
        lines.append(f"Baseline {row.split}: MAE cur_dev_s = {row.mae_cur_dev_s:.2f} с; нулевой прогноз = {row.mae_zero_s:.2f} с, n={row.points}. Официальный score не восстанавливается из приблизительного значения 0.40.")
    failed = tables["checks"].query("failures > 0")
    lines.append("Проверки ключей, склейки, горизонта, таргетов train/test и шаблона: " + (f"есть {len(failed)} нарушенных проверок; см. checks.csv." if len(failed) else "все выполненные проверки прошли (см. checks.csv)."))
    return lines


def write_report(output: Path, tables: dict, manifest: dict, figures: list[dict]) -> None:
    findings = conclusions(tables)
    hashes = {x["path"]: x["sha256"] for x in manifest["input_files"]}
    same = hashes["test/traffic.csv"] == hashes["validate/traffic.csv"]
    overlap = tables["split_overlap"]
    copied = overlap.query("split_a == 'train' and split_b == 'test' and entity == 'packet_id'").iloc[0]
    leakage = (f"Test/validate traffic {'побайтово одинаковы' if same else 'различаются'} (SHA-256). "
        f"Train/test имеют {copied.intersection:,} общих packet_id из {copied.unique_b:,} test. "
        "Прогнозные sample_id и целевые action ID проверяются отдельно в split_overlap.csv. "
        "Нельзя считать эти файлы независимыми временными выборками. Не извлекать будущий факт из расписания другого split. "
        "Отчёт проверяет таргеты только через собственное расписание train/test; ответы validate не реконструируются.")
    method = [
        "Все численные агрегаты рассчитаны по полным файлам. В raincloud ограничена только точечная визуализация (seed=42); KDE и boxplot используют всю соответствующую выборку. Плотность KDE — сглаженная оценка, не новые наблюдения.",
        "NULL означает пустую CSV-ячейку, распознанную pandas как отсутствующее значение; это отличается от location_valid=False. Невалидные даты приводят к ошибке запуска, а не исчезают из анализа.",
        "Пропуски считаются для всех колонок всех девяти входных файлов. UpSet показывает точные, взаимоисключающие комбинации диагностических флагов. Категория без флагов учитывается отдельно.",
        "Свежесть GPS: T минус последний event_time <= T с location_valid=True и заполненными lon/lat. Событие ровно в T−900с не входит в 15-минутное окно. Отсутствие истории не заменено нулём.",
        "Часы на графиках — часы исходной временной шкалы без timezone-конвертации. Источник не содержит offset. event_time, gps_time и receive_time не считаются взаимозаменяемыми.",
        "Когорта real означает числовой packet_id; synthetic_candidate — нечисловой. Это эвристика данной раздачи, не универсальная классификация. Раздельные распределения нужны из-за синтетической аугментации и рассинхронизации часов.",
        "Границы координат: lon ∈ [−180,180], lat ∈ [−90,90]; heading ∈ [0,360). Прохождение этих проверок не подтверждает географическую правдоподобность траектории.",
        "Повторные (ТС,время) считаются как дополнительные строки после первой; разрывы — только положительные интервалы между соседними событиями одного ТС. Повторения не очищаются.",
        "Расписание содержит плановые прибытия, а не справочник физических остановок. В файлах нет route_id/trip_id и статуса дверей: headway и время посадки не вычисляются без допущений.",
        "MAE — среднее по точкам без взвешивания; разрезы имеют разный размер. Небольшой test и общая история ограничивают выводы об обобщении на новые дни. Между когортами не проводился причинный анализ.",
    ]
    actions = [
        "Хранить packet_id как TEXT, action ID отдельно от ID физической остановки, исходные наносекунды без потери точности.",
        "Разделить план и фактические времена; выдавать feature builder только допустимые на T данные. Не вычислять признаки из будущего или из объединённых labels.",
        "Сохранять маски качества, возраст позиции и пропуски. Согласовать fallback при отсутствии свежей телеметрии, а не подставлять скорость 0.",
        "Согласовать временной контракт и отдельно исследовать часы синтетической когорты; случайное разбиение строк не заменяет временную валидацию.",
        "Усилить существующий ML eval: one-to-one sample_id join, полное покрытие, запрет дублей и NaN/Inf. Для локальной оценки использовать MAE, а официальный score брать с платформы.",
    ]
    report = ["# Анализ текущего датасета", "", "Сгенерировано `analyze.py` из локальной раздачи. Версии библиотек и SHA-256 входов — в `manifest.json`. Исходные файлы не изменялись.", "", "## Основные результаты", "", *[f"- {x}" for x in findings], "", "## Пересечения и утечки", "", leakage, "", "## Графики", ""]
    for f in figures:
        report.extend([f"### {f['title']}", "", f"![{f['title']}](figures/{f['file']})", "", f["caption"], ""])
    report.extend(["## Методика и ограничения", "", *[f"- {x}" for x in method], "", "## Что делать дальше", "", *[f"- {x}" for x in actions], "", "## Таблицы", ""])
    for name, table in tables.items():
        report.append(f"- [{name}.csv](tables/{name}.csv): {len(table)} агрегированных строк.")
    (output / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    # Self-contained local page except relative PNG assets; no external scripts, fonts, or network calls.
    parts = ["<!doctype html><html lang='ru'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Анализ датасета транспорта</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#192b38;background:#f5f7f9;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:30px}h2{margin-top:38px}figure{margin:24px 0;background:white;padding:14px}img{width:100%;height:auto}figcaption{font-size:14px;color:#435364}li{margin:8px 0}a{color:#175c84}.note{border-left:4px solid #b46c2a;padding:12px 20px;background:#fff}code{font-size:14px}</style><main><h1>Анализ датасета транспорта</h1><p>Полные файлы, агрегированные результаты. Источники и версии: <a href='manifest.json'>manifest.json</a>.</p><h2>Основные результаты</h2><ul>"]
    parts.extend(f"<li>{html.escape(x)}</li>" for x in findings)
    parts.extend(["</ul><h2>Пересечения и утечки</h2><p class='note'>", html.escape(leakage), "</p><h2>Графики</h2>"])
    for f in figures:
        parts.append(f"<figure><h3>{html.escape(f['title'])}</h3><img src='figures/{html.escape(f['file'])}' alt='{html.escape(f['title'])}'><figcaption>{html.escape(f['caption'])}</figcaption></figure>")
    for title, items in [("Методика и ограничения", method), ("Что делать дальше", actions)]:
        parts.append(f"<h2>{title}</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + "</ul>")
    parts.append("<h2>Агрегированные таблицы</h2><ul>" + "".join(f"<li><a href='tables/{name}.csv'>{name}.csv</a> — {len(table)} строк</li>" for name, table in tables.items()) + "</ul></main></html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("DATASET_DIR", "dataset")))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output.resolve()
    dataset = args.dataset.resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Output and dataset must be separate, non-nested directories")
    frames, tables, manifest = audit(dataset)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / "tables" / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.9g")
    from plots import render_all
    figures = render_all(frames, tables, output / "figures")
    manifest["figures"] = figures
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_report(output, tables, manifest, figures)
    failed = tables["checks"].query("failures > 0")
    print(f"Wrote {len(tables)} aggregate tables, {len(figures)} figures and reports to {output}")
    if len(failed):
        raise SystemExit(f"Audit complete with {len(failed)} failed integrity checks; inspect tables/checks.csv")
    print(f"All {len(tables['checks'])} integrity checks passed")


if __name__ == "__main__":
    main()
