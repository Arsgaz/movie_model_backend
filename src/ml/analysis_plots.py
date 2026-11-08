"""Standalone PNG/SVG figures for descriptive analysis; optional Matplotlib dependency."""

import os
import tempfile
from pathlib import Path

import numpy as np

from ml.analysis import DatasetAnalysis


def render_plots(result: DatasetAnalysis, directory: Path) -> str:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "movie-model-matplotlib"))
    try:
        import matplotlib
    except ImportError as error:
        raise ValueError('Install plotting dependencies: pip install -e ".[ml,analysis]"') from error
    matplotlib.use("Agg")
    from matplotlib import dates, ticker
    from matplotlib import pyplot as plt

    def number(value, _position):
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:g} млн"
        if abs(value) >= 1000:
            return f"{value / 1000:g} тыс."
        return f"{value:g}"

    def style(axis, *, x="", y="", title=""):
        axis.set(xlabel=x, ylabel=y, title=title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    def save(figure, name):
        try:
            figure.savefig(directory / f"{name}.png", dpi=180, facecolor="white")
            figure.savefig(directory / f"{name}.svg", metadata={"Date": None}, facecolor="white")
        finally:
            plt.close(figure)

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titlepad": 14,
            "svg.hashsalt": "movie-model-analysis-v1",
        }
    ):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        values = np.arange(1, 11) / 2
        ax.bar(values, result.rating_counts[1:], width=0.35, color="#326A9F")
        ax.set_xticks(values)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(number))
        ax.set_ylim(0, result.rating_counts[1:].max() * 1.2)
        for value, count in zip(values, result.rating_counts[1:], strict=True):
            ax.annotate(
                f"{count / result.summary['ratings']:.1%}",
                (value, count),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        style(ax, x="Оценка MovieLens", y="Число оценок", title="Распределение оценок")
        save(fig, "01_rating_distribution")

        user_values = np.fromiter(result.user_counts.values(), dtype=np.int64)
        for name, values, title, label in (
            ("02_user_activity", user_values, "Активность пользователей", "Пользователей"),
            (
                "03_movie_popularity",
                result.movie_counts[result.movie_counts > 0],
                "Популярность фильмов по числу оценок",
                "Фильмов",
            ),
        ):
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            edges = np.unique(np.geomspace(max(1, values.min()), max(values.max() + 1, values.min() + 1), 35))
            ax.hist(values, bins=edges, color="#326A9F", edgecolor="white", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            style(
                ax,
                x="Число оценок (логарифмическая шкала)",
                y=f"{label} (логарифмическая шкала)",
                title=title,
            )
            if name == "03_movie_popularity":
                ax.text(
                    0.98,
                    0.95,
                    f"Без оценок в каталоге: {result.summary['movies_without_ratings']:,}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )
            save(fig, name)

        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        for values, label, color in (
            (user_values, "Пользователи", "#D07825"),
            (result.movie_counts, "Фильмы", "#326A9F"),
        ):
            cumulative = np.r_[0, np.cumsum(np.sort(values)[::-1], dtype=np.float64)] / values.sum() * 100
            positions = np.unique(np.linspace(0, len(values), min(1001, len(values) + 1), dtype=int))
            ax.plot(
                positions / len(values) * 100, cumulative[positions], label=label, color=color, linewidth=2
            )
        ax.plot([0, 100], [0, 100], color="#777777", linestyle="--", label="Равномерное распределение")
        ax.set(xlim=(0, 100), ylim=(0, 100))
        ax.legend(loc="lower right")
        style(
            ax,
            x="Доля наиболее активных пользователей / популярных фильмов, %",
            y="Накопленная доля оценок, %",
            title="Концентрация обратной связи",
        )
        save(fig, "04_feedback_concentration")

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
        calendar = np.array(sorted(result.months), dtype="datetime64[M]")
        totals = [result.months[int(month.astype(np.int64))] for month in calendar]
        axes[0].plot(calendar, [value[0] for value in totals], color="#326A9F", linewidth=1.5)
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(number))
        style(axes[0], y="Число оценок", title="Оценки по месяцам UTC")
        axes[1].plot(
            calendar,
            [value[1] / value[0] if value[0] else np.nan for value in totals],
            color="#D07825",
            linewidth=1.5,
        )
        axes[1].set_ylim(0.5, 5)
        style(axes[1], x="Месяц оценки; граничные месяцы могут быть неполными", y="Средняя оценка")
        axes[1].xaxis.set_major_locator(dates.YearLocator(5))
        axes[1].xaxis.set_major_formatter(dates.DateFormatter("%Y"))
        save(fig, "05_rating_timeline")

        if result.genres:
            genres = sorted(result.genres, key=lambda row: (row["movies"], row["genre"]))
            fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout="constrained")
            names = [row["genre"] for row in genres]
            for axis, column, label, color in zip(
                axes, ("movies", "ratings"), ("Фильмов", "Оценок"), ("#326A9F", "#D07825"), strict=True
            ):
                axis.barh(names, [row[column] for row in genres], color=color)
                axis.xaxis.set_major_formatter(ticker.FuncFormatter(number))
                axis.spines[["top", "right"]].set_visible(False)
                axis.set_xlabel(label)
                axis.grid(axis="x", alpha=0.2)
                axis.set_axisbelow(True)
            fig.suptitle("Жанры: фильм может относиться к нескольким категориям", fontsize=13)
            save(fig, "06_genres")

        known = [row for row in result.years if row["year"] is not None]
        if known:
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
            for axis, column, label, color in zip(
                axes, ("movies", "ratings"), ("Фильмов", "Оценок"), ("#326A9F", "#D07825"), strict=True
            ):
                axis.bar(
                    [row["year"] for row in known], [row[column] for row in known], color=color, width=0.9
                )
                axis.yaxis.set_major_formatter(ticker.FuncFormatter(number))
                style(axis, y=label)
            axes[0].set_title(
                f"Годы выпуска из названий; без года: {result.summary['movies_without_release_year']:,}"
            )
            axes[1].set_xlabel("Год выпуска фильма (не год выставления оценки)")
            axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            save(fig, "07_release_years")
    return matplotlib.__version__
