import hashlib
from dataclasses import dataclass

import numpy as np

from ml.evaluation import ranking_metrics
from ml.large_dataset import DiskRatings


@dataclass
class EvaluationCohort:
    profiles: dict[int, dict[int, float]]
    relevant: dict[int, set[int]]
    history_support: dict[int, int]
    all_positive_ids: np.ndarray
    summary: dict


def load_cohort(
    history: DiskRatings, targets: DiskRatings, films, *, max_users=2000, seed=2026
) -> EvaluationCohort:
    if max_users < 1:
        raise ValueError("Evaluation sample must be positive")
    h_user = history.column("user")
    t_user, t_movie, t_value = targets.column("user"), targets.column("movie"), targets.column("value")
    positive = t_value >= 4
    target_users = np.unique(t_user[positive])
    warm = np.intersect1d(np.unique(h_user), target_users)
    selected = (
        warm
        if len(warm) <= max_users
        else np.sort(np.random.default_rng(seed).choice(warm, max_users, replace=False))
    )
    lookup = np.zeros(max(film.movie_id for film in films) + 1, dtype=np.int32)
    for film in films:
        lookup[film.movie_id] = film.tmdb_id
    profiles = {int(user): {} for user in selected}
    truth = {int(user): set() for user in selected}
    # Keep dictionaries for the fixed evaluation cohort only, never all 32M interactions.
    chosen = np.flatnonzero(np.isin(h_user, selected))
    h_movie, h_value = history.column("movie"), history.column("value")
    for index in chosen:
        profiles[int(h_user[index])][int(lookup[h_movie[index]])] = float(h_value[index])
    chosen = np.flatnonzero(positive & np.isin(t_user, selected))
    for index in chosen:
        truth[int(t_user[index])].add(int(lookup[t_movie[index]]))
    positive_ids = lookup[t_movie[positive]]
    movies, frequencies = np.unique(h_movie, return_counts=True)
    support = {int(lookup[movie]): int(count) for movie, count in zip(movies, frequencies, strict=True)}
    return EvaluationCohort(
        profiles,
        truth,
        support,
        positive_ids,
        {
            "eligible_users": len(warm),
            "skipped_users_without_history": len(target_users) - len(warm),
            "evaluation_max_users": max_users,
            "evaluation_seed": seed,
            "evaluation_users_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i4").tobytes()
            ).hexdigest(),
            "target_positive_ratings": int(positive.sum()),
        },
    )


def evaluate_large(
    model,
    history: DiskRatings,
    targets: DiskRatings,
    films,
    *,
    k=10,
    strategy="item_knn",
    max_users=2000,
    seed=2026,
):
    cohort = load_cohort(history, targets, films, max_users=max_users, seed=seed)
    profiles, truth = cohort.profiles, cohort.relevant
    unknown = int(
        np.count_nonzero(~np.isin(cohort.all_positive_ids, np.fromiter(model.by_tmdb, dtype=np.int32)))
    )
    sums = dict.fromkeys(["precision", "recall", "ndcg"], 0.0)
    evaluated, personalized, coverage = 0, 0, set()
    for user in sorted(profiles):
        relevant = truth[int(user)] - profiles[int(user)].keys()
        if not relevant:
            continue
        recommendations, used = model.recommend(profiles[int(user)], limit=k, strategy=strategy)
        ids = [entry.tmdb_id for entry in recommendations]
        for name, value in ranking_metrics(ids, relevant, k).items():
            sums[name] += value
        evaluated += 1
        personalized += used == "item_knn"
        coverage.update(ids)
    return {
        **{f"{name}@{k}": value / evaluated if evaluated else None for name, value in sums.items()},
        **cohort.summary,
        "evaluated_users": evaluated,
        "personalized_users": personalized,
        "unknown_target_positive_ratings": unknown,
        "catalog_coverage": len(coverage) / len(model.films),
    }
