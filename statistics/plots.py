"""Reproducible aggregate figures; no raw identifiers or coordinates are exported."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

COLORS = ["#157f83", "#d88932", "#7156a5"]


def _seconds(df, later, earlier):
    return (pd.to_datetime(df[later], format="mixed") -
            pd.to_datetime(df[earlier], format="mixed")).dt.total_seconds()


def _bool(series):
    return series.astype(str).str.lower().eq("true")


def _raincloud(ax, groups, xlabel, rng):
    """KDE uses all values; the visible rain is a deterministic sample."""
    for i, (label, values) in enumerate(groups):
        x = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        x = x[np.isfinite(x)]
        color = COLORS[i % len(COLORS)]
        if not len(x):
            continue
        if len(x) > 1 and np.ptp(x) > 0:
            grid = np.linspace(x.min(), x.max(), 400)
            density = gaussian_kde(x)(grid)
            ax.fill_between(grid, i + .08, i + .08 + density / density.max() * .30,
                            color=color, alpha=.55, linewidth=0)
        sample = rng.choice(x, min(1000, len(x)), replace=False)
        ax.scatter(sample, i - .16 + rng.uniform(-.055, .055, len(sample)),
                   color=color, alpha=.15, s=6, linewidths=0, rasterized=True)
        q1, med, q3 = np.quantile(x, [.25, .5, .75])
        ax.plot([q1, q3], [i, i], color=color, lw=7, solid_capstyle="round")
        ax.scatter([med], [i], color="white", edgecolor=color, s=34, zorder=5)
    ax.set_yticks(range(len(groups)), [f"{name}\nN={len(values):,}" for name, values in groups])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=.2)
    ax.set_ylim(-.4, len(groups) - .45)


def _quality_flags(df):
    nav = [c for c in ["gps_time", "lon", "lat", "alt", "speed", "heading"] if c in df]
    return pd.DataFrame({
        "GPS невалиден": ~_bool(df.location_valid),
        "NULL в навигации": df[nav].isna().any(axis=1),
        "Скорость >120 км/ч": pd.to_numeric(df.speed, errors="coerce") > 120,
        "receive < event": _seconds(df, "receive_time", "event_time") < 0,
        "gps > event": _seconds(df, "gps_time", "event_time") > 0,
    }, index=df.index)


def _upset(df, split):
    flags = _quality_flags(df)
    counts = flags.value_counts(sort=True, dropna=False)
    # Explicitly retain the all-clear combination, including when its count is zero.
    clean = tuple(False for _ in flags.columns)
    if clean not in counts.index:
        counts.loc[clean] = 0
    combos = list(counts.index)
    n = len(combos)
    fig = plt.figure(figsize=(max(12, n * .65), 8), layout="constrained")
    grid = GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1.5])
    top, bottom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    heights = counts.to_numpy()
    bars = top.bar(np.arange(n), heights, color=["#94a3b8" if not any(c) else COLORS[0] for c in combos])
    top.bar_label(bars, labels=[f"{v:,}" for v in heights], padding=4, fontsize=9)
    top.set_yscale("symlog", linthresh=10)
    top.set_ylabel("Число строк; симметричная лог-шкала")
    top.set_ylim(0, max(10, heights.max()) * 2.5)
    top.set_xticks([])
    top.grid(axis="y", alpha=.15)
    for col, combo in enumerate(combos):
        active = np.flatnonzero(combo)
        bottom.scatter([col] * len(flags.columns), range(len(flags.columns)), color="#e0e5eb", s=44)
        if len(active):
            bottom.plot([col] * len(active), active, color=COLORS[0], lw=2)
            bottom.scatter([col] * len(active), active, color=COLORS[0], s=44, zorder=3)
    bottom.set_yticks(range(len(flags.columns)), flags.columns)
    bottom.invert_yaxis()
    bottom.set_xticks(range(n), ["Нет\nфлагов" if not any(c) else str(i + 1) for i, c in enumerate(combos)])
    bottom.set_xlabel("Точные непересекающиеся комбинации; показаны все наблюдаемые")
    for ax in [top, bottom]:
        ax.set_xlim(-.7, n - .3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Качество телеметрии · {split} · N={len(df):,} · сумма столбцов={int(heights.sum()):,}")
    return fig


def render_all(frames: dict[str, dict[str, pd.DataFrame]], tables: dict[str, pd.DataFrame],
               out: Path) -> list[dict]:
    """Write eight static PNG figures and return their titles and caveats."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "figure.facecolor": "white", "savefig.facecolor": "white"})
    rng = np.random.default_rng(20260925)
    result = []

    def save(fig, file, title, caption):
        fig.savefig(out / file, dpi=160, bbox_inches="tight", pad_inches=.2)
        plt.close(fig)
        result.append({"file": file, "title": title, "caption": caption})

    train = frames["train"]["points"]
    test = frames["test"]["points"]
    cohorts = [("Train · реальные", train[train.cohort.eq("real")]),
               ("Train · синтетика*", train[train.cohort.ne("real")]),
               ("Test · реальные", test)]
    rain_note = ("Источник: labels_train.csv и labels_test.csv. Плотность и квартильный интервал рассчитаны по всем значениям; "
                 "точки — фиксированная случайная выборка до 1000 на группу. Белый маркер — медиана, полоса — Q1–Q3. "
                 "*Синтетическая когорта определена эвристикой нечислового packet_id; validate targets не используются.")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    _raincloud(ax, [(name, d.target_delay_s) for name, d in cohorts], "Фактическое отклонение от расписания, с", rng)
    ax.axvline(0, color="#334155", ls="--", lw=1)
    ax.set_title("Raincloud · распределение целевой задержки")
    save(fig, "01_target_raincloud.png", "Распределение целевой задержки", rain_note)

    fig, axes = plt.subplots(1, 2, figsize=(17, 5), layout="constrained", sharex=True)
    for ax, mode in zip(axes, ["zero", "current"]):
        values = [(name, d.target_delay_s.abs() if mode == "zero" else (d.target_delay_s - d.cur_dev_s).abs()) for name, d in cohorts]
        _raincloud(ax, values, "Абсолютная ошибка, с", rng)
        ax.set_title("Прогноз = 0" if mode == "zero" else "Прогноз = cur_dev_s")
    fig.suptitle("Raincloud · ошибки двух baseline")
    save(fig, "02_baseline_raincloud.png", "Распределение ошибок baseline", rain_note + " Ошибки вычислены отдельно для каждой точки; это описательная оценка, а не независимый эксперимент.")

    for i, split in enumerate(["train", "test"], 3):
        save(_upset(frames[split]["traffic"], split), f"0{i}_quality_upset_{split}.png",
             f"Точные пересечения флагов качества · {split}",
             f"Источник: {split}/traffic.csv. Все наблюдаемые комбинации включены, без отсечения редких; каждая строка входит ровно в один столбец. "
             "NULL в навигации — пропуск хотя бы одного из gps_time/lon/lat/alt/speed/heading. Порог скорости 120 км/ч — диагностическая эвристика, а не физический предел. "
             "Столбец без флагов не означает доказанную корректность данных. Test и validate идентичны по телеметрии.")

    cols = [c for c in frames["train"]["traffic"].columns if c != "cohort"]
    matrix = np.array([frames[s]["traffic"].reindex(columns=cols).isna().mean().to_numpy() * 100 for s in frames])
    fig, ax = plt.subplots(figsize=(15, 4), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1, float(matrix.max())))
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(frames)), list(frames))
    for row in range(len(frames)):
        for col in range(len(cols)):
            ax.text(col, row, f"{matrix[row, col]:.2f}%", ha="center", va="center", fontsize=8,
                    color="white" if matrix[row, col] > matrix.max() * .6 else "#172033")
    fig.colorbar(image, ax=ax, label="Доля NULL, %", shrink=.8)
    ax.set_title("Пропущенные значения во всех исходных колонках телеметрии")
    save(fig, "05_missing_heatmap.png", "NULL по колонкам и выборкам",
         "Источник: train/test/validate traffic.csv. Доля pandas.isna() от всех строк соответствующего split; служебная добавленная колонка cohort исключена. Невалидный GPS при заполненных координатах не считается NULL.")

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for color, (split, bundle) in zip(COLORS, frames.items()):
        ages = pd.to_numeric(bundle["points"].valid_ping_age_s, errors="coerce")
        finite = np.sort(ages[np.isfinite(ages)].to_numpy())
        if len(finite):
            ax.step(np.log1p(finite), np.arange(1, len(finite) + 1) / len(finite), where="post", color=color,
                    label=f"{split}: с историей {len(finite):,}; без истории {ages.isna().sum():,}")
    ax.axvline(np.log1p(900), color="#334155", ls="--", label="15 минут")
    ticks = np.array([0, 10, 60, 300, 900, 3600, 21600, 86400])
    ax.set_xticks(np.log1p(ticks), ["0", "10", "60", "300", "900", "3 600", "21 600", "86 400"])
    ax.set_xlabel("Возраст последнего валидного GPS на момент T, с · ось log(1+x)")
    ax.set_ylabel("Доля точек среди имеющих прошлый валидный GPS")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.2)
    ax.legend(loc="lower right")
    ax.set_title("ECDF · свежесть доступной телеметрии в прогнозных точках")
    save(fig, "06_gps_age_ecdf.png", "Свежесть GPS на момент прогноза",
         "Источник: points/labels и traffic каждого split; только валидный GPS с event_time ≤ T. ECDF условная: точки без прошлого валидного GPS исключены из знаменателя и отдельно посчитаны в легенде. Receive_time здесь не ограничивает доступность. Ось преобразована log(1+x), подписи в секундах.")

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), layout="constrained", sharex=True)
    for ax, color, (split, bundle) in zip(axes, COLORS, frames.items()):
        counts = pd.to_datetime(bundle["traffic"].event_time, format="mixed").dt.floor("h").value_counts().sort_index()
        ax.bar(counts.index, counts.values, width=.037, color=color)
        ax.set_ylabel(f"{split}\nСтрок/час")
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xlabel("event_time из CSV; часовой пояс источником явно не задан")
    fig.suptitle("Почасовой объём телеметрии · split показаны отдельно")
    fig.autofmt_xdate(rotation=25)
    save(fig, "07_hourly_volume.png", "Почасовой объём телеметрии",
         "Источник: traffic.csv каждого split; часы округлены вниз по event_time, без изменения временной зоны. Split нельзя суммировать как независимые наблюдения: test/validate traffic идентичны, их packet_id входят в train. Синтетические временные метки train выходят за границы суток.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    tr = frames["train"]["traffic"]
    for ax, later, earlier, title in [(axes[0], "receive_time", "event_time", "Получение − событие"),
                                      (axes[1], "gps_time", "event_time", "GPS − событие")]:
        for color, cohort, label in zip(COLORS, ["real", "synthetic_candidate"], ["Train · реальные", "Train · синтетика*"]):
            x = _seconds(tr[tr.cohort.eq(cohort)], later, earlier).dropna().sort_values().to_numpy()
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=f"{label} · N={len(x):,}", color=color)
        ax.set_xscale("symlog", linthresh=1)
        ax.axvline(0, color="#334155", ls="--", lw=1)
        ax.set_xlabel("Разность времён, с · symlog, линейно в ±1 с")
        ax.set_ylabel("Накопленная доля непустых значений")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("ECDF · согласованность часов реальной и синтетической телеметрии")
    save(fig, "08_clock_ecdf.png", "Согласованность временных меток",
         "Источник: train/traffic.csv. ECDF использует все ненулевые разности, без отсечения выбросов. Отрицательная разность означает, что левая временная метка раньше правой. *Эвристика синтетики — нечисловой packet_id. Расхождения требуют отдельного контракта обработки времени.")
    return result
