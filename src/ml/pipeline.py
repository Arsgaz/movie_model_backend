import gc
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from ml.dataset import digest, load_dataset, write_json
from ml.evaluation import chronological_split, evaluate
from ml.model import ItemModel, Parameters


def train(dataset_path: Path, output: Path, *, evaluation_users: int = 2000, block_size: int = 128) -> dict:
    if evaluation_users < 1 or block_size < 1:
        raise ValueError("Evaluation user limit and block size must be positive")
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory")
    dataset = load_dataset(dataset_path)
    train_rows, validation, test = chronological_split(dataset.ratings)
    from ml.large_dataset import DiskRatings

    large = isinstance(dataset.ratings, DiskRatings)

    def measure(model, history, targets, *, strategy="item_knn"):
        if large:
            from ml.large_evaluation import evaluate_large

            return evaluate_large(
                model, history, targets, dataset.films, strategy=strategy, max_users=evaluation_users
            )
        return evaluate(model, history, targets, dataset.films, strategy=strategy)

    def fit(rows, parameters):
        print(f"Fitting {len(rows):,} ratings: {asdict(parameters)}", flush=True)
        return ItemModel.fit(
            dataset.films,
            rows,
            parameters,
            block_size=block_size,
            progress=lambda message: print(message, flush=True),
        )

    if large:
        # Both neighbor budgets share exactly the same underlying similarity computation.
        grid = [
            Parameters(neighbors=n, shrinkage=50, min_support=3, max_user_positives=100) for n in (40, 80)
        ]
        base = fit(train_rows, grid[-1])
        matrix = base.similarities
        positions = np.arange(matrix.nnz) - np.repeat(matrix.indptr[:-1], np.diff(matrix.indptr))
        keep = positions < grid[0].neighbors
        lengths = np.minimum(np.diff(matrix.indptr), grid[0].neighbors)
        from scipy import sparse

        smaller = sparse.csr_matrix(
            (matrix.data[keep], matrix.indices[keep], np.r_[0, np.cumsum(lengths)]), shape=matrix.shape
        )
        models = [ItemModel(base.films, smaller, base.popularity, grid[0]), base]
        del positions, keep, lengths
    else:
        grid = [Parameters(neighbors=n, shrinkage=s) for n in (40, 80) for s in (10.0, 50.0)]
        models = [fit(train_rows, parameters) for parameters in grid]
    trials = []
    for parameters, model in zip(grid, models, strict=True):
        metrics = measure(model, train_rows, validation)
        print(f"Validation {parameters.neighbors} neighbors: {metrics['ndcg@10']}", flush=True)
        trials.append({"parameters": asdict(parameters), "metrics": metrics})
    baseline = measure(models[0], train_rows, validation, strategy="popularity")
    if baseline["evaluated_users"] == 0:
        raise ValueError("No validation users with history; model selection is impossible")
    best = max(range(len(trials)), key=lambda i: trials[i]["metrics"]["ndcg@10"])
    selected = "item_knn" if trials[best]["metrics"]["ndcg@10"] > baseline["ndcg@10"] else "popularity"
    parameters = grid[best]
    # Freeze hyperparameters before accessing test. Test model may now use validation history.
    train_validation = (
        dataset.ratings.before(int(test.column("time").min())) if large else train_rows + validation
    )
    del models
    if large:
        del base, smaller, matrix
    gc.collect()
    test_model = fit(train_validation, parameters)
    test_baseline = measure(test_model, train_validation, test, strategy="popularity")
    test_selected = measure(test_model, train_validation, test, strategy=selected)
    print(
        f"Test NDCG: selected={test_selected['ndcg@10']}, popularity={test_baseline['ndcg@10']}", flush=True
    )
    if not test_selected["evaluated_users"]:
        raise ValueError("No test users with history")
    report = {
        "protocol": "global_temporal_70_15_15",
        "k": 10,
        "relevant_rating_min": 4,
        "selection_metric": "validation_ndcg@10",
        "evaluation_user_limit": evaluation_users if large else None,
        "cooccurrence_block_size": block_size,
        "selected_strategy": selected,
        "selected_parameters": asdict(parameters),
        "split": {
            "train": len(train_rows),
            "validation": len(validation),
            "test": len(test),
            "validation_from_timestamp": int(validation.column("time").min())
            if large
            else validation[0].timestamp,
            "test_from_timestamp": int(test.column("time").min()) if large else test[0].timestamp,
        },
        "validation": {"popularity": baseline, "item_knn_trials": trials},
        "test": {"popularity": test_baseline, "selected": test_selected},
        "dataset": dataset.manifest,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    # Final serving model uses all observations only AFTER untouched-test measurement.
    del test_model
    gc.collect()
    final = fit(dataset.ratings, parameters)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", report)
    source_readme = dataset_path / "SOURCE_README.txt"
    (output / "SOURCE_README.txt").write_bytes(source_readme.read_bytes())
    metadata = {
        "model_version": output.name,
        "created_at": datetime.now(UTC).isoformat(),
        "strategy": selected,
        "dataset_sha256": dataset.manifest["source_sha256"],
        "prepared_manifest_sha256": digest(dataset_path / "manifest.json"),
        "training_ratings": len(dataset.ratings),
        "training_users": dataset.manifest["users"],
        "metrics_sha256": digest(output / "metrics.json"),
        "versions": report["versions"],
        "source_readme_sha256": digest(output / "SOURCE_README.txt"),
    }
    final.save(output / "model", metadata)
    write_json(
        output / "summary.json",
        {
            "model_version": output.name,
            "strategy": selected,
            "validation_ndcg@10": trials[best]["metrics"]["ndcg@10"]
            if selected == "item_knn"
            else baseline["ndcg@10"],
            "test": report["test"],
            "movies": len(final.films),
            "ratings": len(dataset.ratings),
        },
    )
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))
