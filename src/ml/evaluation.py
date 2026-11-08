import math
from collections import defaultdict
from collections.abc import Sequence

from ml.dataset import Film, Observation
from ml.model import ItemModel


def chronological_split(ratings: Sequence[Observation]) -> tuple[Sequence[Observation], ...]:
    from ml.large_dataset import DiskRatings

    if isinstance(ratings, DiskRatings):
        import numpy as np

        time = ratings.column("time")
        if len(time) < 10:
            raise ValueError("Need at least ten observations for a three-way split")
        first_index, second_index = int(len(time) * 0.7), int(len(time) * 0.85)
        partitioned = np.partition(time, (first_index, second_index))
        first, second = int(partitioned[first_index]), int(partitioned[second_index])
        del partitioned
        parts = ratings.before(first), ratings.between(first, second), ratings.between(second, None)
        if not all(len(part) for part in parts):
            raise ValueError("Timestamp distribution cannot support the chronological split")
        return parts
    ordered = sorted(ratings, key=lambda row: (row.timestamp, row.user_id, row.movie_id))
    if len(ordered) < 10:
        raise ValueError("Need at least ten observations for a three-way split")
    # Global cutoffs keep every user's future out of training, including timestamp ties.
    first = ordered[int(len(ordered) * 0.7)].timestamp
    second = ordered[int(len(ordered) * 0.85)].timestamp
    train = tuple(row for row in ordered if row.timestamp < first)
    validation = tuple(row for row in ordered if first <= row.timestamp < second)
    test = tuple(row for row in ordered if row.timestamp >= second)
    if not train or not validation or not test:
        raise ValueError("Timestamp distribution cannot support the chronological split")
    return train, validation, test


def ranking_metrics(recommended: Sequence[int], relevant: set[int], k: int) -> dict[str, float]:
    if not relevant or k < 1 or len(recommended) != len(set(recommended)):
        raise ValueError("Metrics need a positive target set and distinct recommendations")
    hits = [int(movie in relevant) for movie in recommended[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"precision": sum(hits) / k, "recall": sum(hits) / len(relevant), "ndcg": dcg / ideal}


def evaluate(
    model: ItemModel,
    history: Sequence[Observation],
    targets: Sequence[Observation],
    films: Sequence[Film],
    *,
    k: int = 10,
    strategy: str = "item_knn",
) -> dict:
    tmdb = {film.movie_id: film.tmdb_id for film in films}
    profiles: dict[int, dict[int, float]] = defaultdict(dict)
    for row in history:
        profiles[row.user_id][tmdb[row.movie_id]] = row.rating
    relevant: dict[int, set[int]] = defaultdict(set)
    for row in targets:
        if row.rating >= 4:
            relevant[row.user_id].add(tmdb[row.movie_id])
    sums = dict.fromkeys(["precision", "recall", "ndcg"], 0.0)
    evaluated, personalized, skipped_no_history, unknown_targets, all_targets = 0, 0, 0, 0, 0
    recommended_catalog: set[int] = set()
    for user in sorted(relevant):
        truth = relevant[user]
        all_targets += len(truth)
        unknown_targets += sum(movie not in model.by_tmdb for movie in truth)
        # New profiles with no known positive seed use the product's popularity fallback.
        # Entirely new users are reported separately; this benchmark measures warm users.
        if user not in profiles:
            skipped_no_history += 1
            continue
        history_values = profiles[user]
        truth = truth - history_values.keys()
        if not truth:
            continue
        recommendations, used = model.recommend(history_values, limit=k, strategy=strategy)
        ids = [entry.tmdb_id for entry in recommendations]
        for key, value in ranking_metrics(ids, truth, k).items():
            sums[key] += value
        evaluated += 1
        personalized += used == "item_knn"
        recommended_catalog.update(ids)
    return {
        **{f"{key}@{k}": value / evaluated if evaluated else None for key, value in sums.items()},
        "evaluated_users": evaluated,
        "personalized_users": personalized,
        "skipped_users_without_history": skipped_no_history,
        "target_positive_ratings": all_targets,
        "unknown_target_positive_ratings": unknown_targets,
        "catalog_coverage": len(recommended_catalog) / len(model.films),
    }
