"""Diagnostic replay of a frozen temporal experiment, without touching the serving model."""

import csv
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from ml.bootstrap import paired_bootstrap
from ml.dataset import digest, load_dataset, write_json
from ml.evaluation import chronological_split, ranking_metrics
from ml.large_dataset import DiskRatings
from ml.large_evaluation import EvaluationCohort, load_cohort
from ml.model import ItemModel, Parameters

METRICS = ("precision", "recall", "ndcg")
HISTORY_GROUPS = ("1-19", "20-99", "100-499", "500+")
ITEM_GROUPS = ("unseen", "tail_1-49", "middle_50-999", "head_1000+")


def history_group(size: int) -> str:
    return "1-19" if size < 20 else "20-99" if size < 100 else "100-499" if size < 500 else "500+"


def item_group(support: int) -> str:
    return (
        "unseen"
        if support == 0
        else "tail_1-49"
        if support < 50
        else "middle_50-999"
        if support < 1000
        else "head_1000+"
    )


def evaluate_cohort(model: ItemModel, cohort: EvaluationCohort, *, strategy: str, k: int = 10) -> dict:
    rows, group_rows, targets, recommendations, users = [], [], [], [], []
    for user in sorted(cohort.profiles):
        profile = cohort.profiles[user]
        truth = cohort.relevant[user] - profile.keys()
        if not truth:
            continue
        group = history_group(len(profile))
        users.append(
            {
                "user_id": user,
                "history_ratings": len(profile),
                "history_group": group,
                "relevant_targets": len(truth),
                "known_history_ratings": sum(movie in model.by_tmdb for movie in profile),
            }
        )
        partition = {name: set() for name in ITEM_GROUPS}
        for movie in sorted(truth):
            support = cohort.history_support.get(movie, 0)
            name = item_group(support)
            partition[name].add(movie)
            targets.append(
                {"user_id": user, "tmdb_id": movie, "history_support": support, "item_group": name}
            )
        for label, algorithm in (("popularity", "popularity"), ("selected", strategy)):
            items, used = model.recommend(profile, limit=k, strategy=algorithm)
            ids = [item.tmdb_id for item in items]
            metrics = ranking_metrics(ids, truth, k)
            rows.append(
                {"user_id": user, "strategy": label, "used_strategy": used, "history_group": group, **metrics}
            )
            for rank, item in enumerate(items, 1):
                recommendations.append(
                    {
                        "user_id": user,
                        "strategy": label,
                        "rank": rank,
                        "tmdb_id": item.tmdb_id,
                        "hit": int(item.tmdb_id in truth),
                        "item_group": item_group(cohort.history_support.get(item.tmdb_id, 0)),
                    }
                )
            # Original top-k stays unchanged. Only the relevant target set is restricted to each group.
            for name, relevant in partition.items():
                if relevant:
                    group_rows.append(
                        {
                            "user_id": user,
                            "strategy": label,
                            "item_group": name,
                            "relevant_targets": len(relevant),
                            **ranking_metrics(ids, relevant, k),
                        }
                    )
    if not users:
        raise ValueError("No evaluable users in the saved cohort")
    return {
        "per_user": rows,
        "per_item_group": group_rows,
        "cohort": users,
        "targets": targets,
        "recommendations": recommendations,
    }


def summarize_pairs(rows: list[dict], *, resamples: int, seed: int) -> dict:
    pairs: dict[int, dict[str, dict]] = {}
    for row in rows:
        pair = pairs.setdefault(row["user_id"], {})
        if row["strategy"] in pair:
            raise ValueError("Duplicate strategy for a user")
        pair[row["strategy"]] = row
    if not pairs:
        return {"users": 0, "small_group": True, "metrics": {}}
    if any(set(pair) != {"popularity", "selected"} for pair in pairs.values()):
        raise ValueError("Bootstrap must compare the same users in both strategies")
    return {
        "users": len(pairs),
        "small_group": len(pairs) < 30,
        "metrics": {
            metric: paired_bootstrap(
                np.array(
                    [
                        [pairs[user][name][metric] for name in ("popularity", "selected")]
                        for user in sorted(pairs)
                    ]
                ),
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        },
    }


def summarize_details(details: dict, *, resamples: int, seed: int) -> dict:
    return {
        "overall": summarize_pairs(details["per_user"], resamples=resamples, seed=seed),
        "by_history": {
            group: summarize_pairs(
                [row for row in details["per_user"] if row["history_group"] == group],
                resamples=resamples,
                seed=seed,
            )
            for group in HISTORY_GROUPS
        },
        "by_item_support": {
            group: summarize_pairs(
                [row for row in details["per_item_group"] if row["item_group"] == group],
                resamples=resamples,
                seed=seed,
            )
            for group in ITEM_GROUPS
        },
    }


def write_records(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evaluation table without a schema")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assess(
    dataset_path: Path,
    training_run: Path,
    output: Path,
    *,
    resamples: int = 2000,
    seed: int = 2026,
    block_size: int = 128,
) -> dict:
    if output.exists() or resamples < 100 or block_size < 1:
        raise ValueError("Use a new output directory, positive block size and at least 100 resamples")
    source_metrics = training_run / "metrics.json"
    source_model_manifest = training_run / "model/manifest.json"
    experiment = json.loads(source_metrics.read_text(encoding="utf-8"))
    saved_model = json.loads(source_model_manifest.read_text(encoding="utf-8"))
    if digest(source_metrics) != saved_model["metrics_sha256"]:
        raise ValueError("Saved training metrics checksum mismatch")
    if digest(dataset_path / "manifest.json") != saved_model["prepared_manifest_sha256"]:
        raise ValueError("Prepared dataset does not match the frozen training run")
    if experiment["protocol"] != "global_temporal_70_15_15":
        raise ValueError("Unsupported frozen evaluation protocol")
    parameters = Parameters(**experiment["selected_parameters"])
    if (
        asdict(parameters) != saved_model["parameters"]
        or experiment["selected_strategy"] != saved_model["strategy"]
    ):
        raise ValueError("Frozen parameters differ from the serving manifest")
    dataset = load_dataset(dataset_path)
    if not isinstance(dataset.ratings, DiskRatings):
        raise ValueError("Diagnostic replay currently requires disk-backed prepared format 2")
    train, validation, test = chronological_split(dataset.ratings)
    first, second = int(validation.column("time").min()), int(test.column("time").min())
    recorded_split = experiment["split"]
    if (
        first != recorded_split["validation_from_timestamp"]
        or second != recorded_split["test_from_timestamp"]
        or [len(train), len(validation), len(test)]
        != [recorded_split[key] for key in ("train", "validation", "test")]
    ):
        raise ValueError("Replayed temporal split differs from the saved experiment")
    history = dataset.ratings.before(second)
    recorded = experiment["test"]["selected"]
    cohort = load_cohort(
        history,
        test,
        dataset.films,
        max_users=recorded["evaluation_max_users"],
        seed=recorded["evaluation_seed"],
    )
    if cohort.summary["evaluation_users_sha256"] != recorded["evaluation_users_sha256"]:
        raise ValueError("Replayed user cohort differs from the saved experiment")
    print(
        f"Rebuilding frozen test model on {len(history):,} historical ratings; "
        "serving artifact stays unchanged",
        flush=True,
    )
    model = ItemModel.fit(
        dataset.films,
        history,
        parameters,
        block_size=block_size,
        progress=lambda message: print(message, flush=True),
    )
    details = evaluate_cohort(model, cohort, strategy=experiment["selected_strategy"], k=experiment["k"])
    if len(details["cohort"]) != recorded["evaluated_users"]:
        raise ValueError("Replayed evaluated user count differs from the saved experiment")
    summary = summarize_details(details, resamples=resamples, seed=seed)
    for strategy, key in (("popularity", "popularity"), ("selected", "selected")):
        for metric in METRICS:
            actual = summary["overall"]["metrics"][metric][strategy]["mean"]
            if not np.isclose(
                actual, experiment["test"][key][f"{metric}@{experiment['k']}"], rtol=0, atol=1e-10
            ):
                raise ValueError("Replayed aggregate metrics differ from the saved experiment")
    summary.update(
        {
            "protocol": experiment["protocol"],
            "model_version": saved_model["model_version"],
            "selected_strategy": experiment["selected_strategy"],
            "k": experiment["k"],
            "parameters": asdict(parameters),
            "cohort": cohort.summary,
            "test_model_training_ratings": len(history),
            "test_model_movies": len(model.films),
            "bootstrap_seed": seed,
            "bootstrap_unit": "user",
            "bootstrap_method": "paired_percentile",
            "thresholds": {"history_ratings": [20, 100, 500], "historical_item_ratings": [1, 50, 1000]},
            "scope": "diagnostic_replay_of_frozen_collaborative_model_not_hybrid",
            "notes": [
                "Intervals condition on the frozen model and observed warm-user cohort.",
                "They do not include uncertainty from model fitting or temporal shifts.",
                "Segment intervals are exploratory, without multiple-comparison correction.",
                "Item-group metrics restrict targets only, never the full-catalog top-k.",
                "Unknown positive targets remain misses; cold users are excluded and counted.",
                "No parameters are selected from this diagnostic replay.",
            ],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".quality-") as temporary:
        work = Path(temporary)
        write_json(work / "summary.json", summary)
        for name, rows in details.items():
            write_records(work / f"{name}.csv", rows)
        flat = []
        for dimension, groups in (
            ("overall", {"all": summary["overall"]}),
            ("history", summary["by_history"]),
            ("item_support", summary["by_item_support"]),
        ):
            for group, result in groups.items():
                for metric, estimate in result["metrics"].items():
                    for strategy in ("popularity", "selected", "paired_difference"):
                        flat.append(
                            {
                                "dimension": dimension,
                                "group": group,
                                "metric": f"{metric}@{summary['k']}",
                                "strategy": strategy,
                                "users": result["users"],
                                "small_group": result["small_group"],
                                **{field: estimate[strategy][field] for field in ("mean", "low", "high")},
                            }
                        )
        write_records(work / "intervals.csv", flat)
        from ml.quality_plots import render_quality_plots

        version = render_quality_plots(summary, work)
        manifest = {
            "format_version": 1,
            "artifact": "model_quality_evaluation",
            "training_metrics_sha256": digest(source_metrics),
            "source_model_manifest_sha256": digest(source_model_manifest),
            "prepared_manifest_sha256": digest(dataset_path / "manifest.json"),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": version,
            },
            "files": {path.name: digest(path) for path in sorted(work.iterdir())},
        }
        write_json(work / "manifest.json", manifest)
        work.rename(output)
    return summary
