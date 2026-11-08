import argparse
import json
from dataclasses import asdict
from pathlib import Path

from config.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens offline recommendation pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download_parser = commands.add_parser("download")
    download_parser.add_argument("--source", choices=["small", "32m"], default="32m")
    download_parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "data/movielens")
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", choices=["small", "32m"], default="32m")
    prepare_parser.add_argument("--archive", type=Path)
    prepare_parser.add_argument("--output", type=Path)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/movielens/prepared-32m-v1")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--evaluation-users", type=int, default=2000)
    train_parser.add_argument("--block-size", type=int, default=128)
    train_parser.add_argument(
        "--k", type=int, default=30, help="Ranking cutoff for selection and evaluation (1–100)"
    )
    analysis_parser = commands.add_parser("analyze")
    analysis_parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/movielens/prepared-32m-v1"
    )
    analysis_parser.add_argument("--output", type=Path, required=True)
    analysis_parser.add_argument("--chunk-size", type=int, default=1_000_000)
    quality_parser = commands.add_parser("assess")
    quality_parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/movielens/prepared-32m-v1"
    )
    quality_parser.add_argument("--training-run", type=Path, required=True)
    quality_parser.add_argument("--output", type=Path, required=True)
    quality_parser.add_argument("--resamples", type=int, default=2000)
    quality_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    quality_parser.add_argument("--block-size", type=int, default=128)
    recommend_parser = commands.add_parser("recommend")
    recommend_parser.add_argument("--model", type=Path, required=True)
    recommend_parser.add_argument(
        "--ratings", default="{}", help="TMDB movie IDs to ratings, e.g. '{\"603\":5}'"
    )
    recommend_parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    try:
        if args.command == "download":
            from ml.dataset import download

            if args.source == "32m":
                from ml.large_dataset import download_large

                result = {"archive": str(download_large(args.directory))}
            else:
                result = {"archive": str(download(args.directory))}
        elif args.command == "prepare":
            from ml.dataset import prepare

            if args.source == "32m":
                from ml.large_dataset import prepare_large

                result = prepare_large(
                    args.archive or PROJECT_ROOT / "data/movielens/ml-32m.zip",
                    args.output or PROJECT_ROOT / "data/movielens/prepared-32m-v1",
                )
            else:
                result = prepare(
                    args.archive or PROJECT_ROOT / "data/movielens/ml-latest-small.zip",
                    args.output or PROJECT_ROOT / "data/movielens/prepared-v1",
                )
        elif args.command == "analyze":
            from ml.analysis import analyze

            result = analyze(args.dataset, args.output, chunk_size=args.chunk_size)
        elif args.command == "assess":
            from ml.quality_evaluation import assess

            result = assess(
                args.dataset,
                args.training_run,
                args.output,
                resamples=args.resamples,
                seed=args.bootstrap_seed,
                block_size=args.block_size,
            )
            result = {
                "output": str(args.output),
                "model_version": result["model_version"],
                "overall": result["overall"],
                "history_groups": {name: value["users"] for name, value in result["by_history"].items()},
                "item_groups": {name: value["users"] for name, value in result["by_item_support"].items()},
            }
        elif args.command == "train":
            from ml.pipeline import train

            result = train(
                args.dataset,
                args.output,
                evaluation_users=args.evaluation_users,
                block_size=args.block_size,
                k=args.k,
            )
        else:
            from ml.model import ItemModel

            model, manifest = ItemModel.load(args.model)
            profile = {int(key): float(value) for key, value in json.loads(args.ratings).items()}
            recommendations, used = model.recommend(profile, limit=args.limit, strategy=manifest["strategy"])
            result = {
                "strategy": used,
                "model_version": manifest["model_version"],
                "items": [asdict(item) for item in recommendations],
            }
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
