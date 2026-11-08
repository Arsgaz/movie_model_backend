import os
import tempfile
from pathlib import Path

from ml.quality_evaluation import HISTORY_GROUPS, ITEM_GROUPS, METRICS


def render_quality_plots(summary: dict, directory: Path) -> str:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "movie-model-matplotlib"))
    try:
        import matplotlib
    except ImportError as error:
        raise ValueError('Install plotting dependencies: pip install -e ".[ml,analysis]"') from error
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    k = summary["k"]
    labels = {"popularity": "Популярность", "selected": "Выбранная модель"}
    colors = {"popularity": "#D07825", "selected": "#326A9F"}

    def point(axis, x, estimate, color, label=None):
        axis.plot([x, x], [estimate["low"], estimate["high"]], color=color, linewidth=2)
        axis.plot([x - 0.025, x + 0.025], [estimate["low"]] * 2, color=color)
        axis.plot([x - 0.025, x + 0.025], [estimate["high"]] * 2, color=color)
        axis.plot(x, estimate["mean"], "o", color=color, label=label, markersize=6)

    def style(axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)

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
            "axes.titlesize": 12,
            "svg.hashsalt": "movie-model-quality-v1",
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
        for axis, metric in zip(axes, METRICS, strict=True):
            values = summary["overall"]["metrics"][metric]
            for index, strategy in enumerate(("popularity", "selected")):
                point(axis, index, values[strategy], colors[strategy])
            axis.set_xticks([0, 1], [labels[name] for name in ("popularity", "selected")], rotation=12)
            axis.set(xlim=(-0.4, 1.4), title=f"{metric.upper()}@{k}", ylabel="Значение метрики")
            style(axis)
        fig.suptitle(f"Среднее и 95 % bootstrap-интервал; пользователей: {summary['overall']['users']}")
        save(fig, "01_overall_intervals")

        fig, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
        for index, group in enumerate(HISTORY_GROUPS):
            result = summary["by_history"][group]
            for shift, strategy in ((-0.1, "popularity"), (0.1, "selected")):
                if result["users"]:
                    point(
                        axis,
                        index + shift,
                        result["metrics"]["ndcg"][strategy],
                        colors[strategy],
                        labels[strategy] if index == 0 else None,
                    )
        axis.set_xticks(
            range(len(HISTORY_GROUPS)),
            [f"{group}\nn={summary['by_history'][group]['users']}" for group in HISTORY_GROUPS],
        )
        axis.set(
            xlabel="Число оценок до тестовой границы",
            ylabel=f"NDCG@{k}",
            title="Качество по длине истории: среднее и 95 % интервал",
        )
        style(axis)
        # Explicit handles keep the legend correct even when the first group has no observations.
        from matplotlib.lines import Line2D

        axis.legend(
            handles=[
                Line2D([0], [0], marker="o", color=colors[name], label=labels[name])
                for name in ("popularity", "selected")
            ]
        )
        save(fig, "02_history_segments")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
        group_labels = {
            "unseen": "Нет в истории",
            "tail_1-49": "1–49",
            "middle_50-999": "50–999",
            "head_1000+": "1000+",
        }
        for axis, metric in zip(axes, ("recall", "ndcg"), strict=True):
            for index, group in enumerate(ITEM_GROUPS):
                result = summary["by_item_support"][group]
                for shift, strategy in ((-0.1, "popularity"), (0.1, "selected")):
                    if result["users"]:
                        point(axis, index + shift, result["metrics"][metric][strategy], colors[strategy])
            axis.set_xticks(
                range(len(ITEM_GROUPS)),
                [
                    f"{group_labels[group]}\nn={summary['by_item_support'][group]['users']}"
                    for group in ITEM_GROUPS
                ],
            )
            axis.set(xlabel="Число оценок фильма в истории до теста", ylabel=f"{metric.upper()}@{k}")
            style(axis)
        axes[0].legend(
            handles=[
                Line2D([0], [0], marker="o", color=colors[name], label=labels[name])
                for name in ("popularity", "selected")
            ]
        )
        fig.suptitle(
            "Качество по группам целевых фильмов\n"
            "n — пользователей с целями группы; top-k не меняется; 95 % интервалы"
        )
        save(fig, "03_item_segments")

        fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), layout="constrained")
        for axis, metric in zip(axes, METRICS, strict=True):
            value = summary["overall"]["metrics"][metric]["paired_difference"]
            axis.axhline(0, color="#777777", linestyle="--")
            point(axis, 0, value, "#326A9F")
            axis.set(
                xticks=[],
                xlim=(-0.5, 0.5),
                title=f"Δ {metric.upper()}@{k}",
                ylabel="Модель минус популярность",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle("Парная разность на тех же пользователях: среднее и 95 % интервал")
        save(fig, "04_paired_differences")
    return matplotlib.__version__
