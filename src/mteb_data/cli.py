"""CLI for auditing and building MTEB training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mteb_data.catalog import audit_catalog, load_catalog
from mteb_data.pipeline import build_dataset
from mteb_data.sources.external import load_external_sources
from mteb_data.sources.mteb import EvaluationSplitError, assert_trainable_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mteb-data", description="MTEB Multilingual v2 training-data pipeline")
    parser.add_argument("--catalog", type=Path, default=None, help="Path to mmteb_multilingual_v2.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="Summarize eligible/ineligible tasks")

    p_ext = sub.add_parser("external", help="Show prioritized external sources")
    p_ext.add_argument("--config", type=Path, default=None)

    p_check = sub.add_parser("check-split", help="Validate whether a split is trainable")
    p_check.add_argument("--task", required=True)
    p_check.add_argument("--split", required=True)

    p_build = sub.add_parser("build", help="Download eligible splits and generate training views")
    p_build.add_argument("--output", type=Path, default=Path("data/processed"))
    p_build.add_argument("--families", nargs="*", default=None)
    p_build.add_argument("--tasks", nargs="*", default=None)
    p_build.add_argument("--max-rows-per-split", type=int, default=None)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--pairs-per-anchor", type=int, default=1)
    p_build.add_argument("--no-cross-language", action="store_true")

    args = parser.parse_args(argv)
    catalog = load_catalog(args.catalog)

    if args.command == "audit":
        print(json.dumps(audit_catalog(catalog), indent=2))
        return 0

    if args.command == "external":
        sources = load_external_sources(args.config)
        payload = [
            {
                "name": s.name,
                "dataset": s.dataset,
                "priority": s.priority,
                "overlap_risk": s.overlap_risk,
                "default_cap": s.default_cap,
                "license": s.license,
                "description": s.description,
                "notes": s.notes,
            }
            for s in sources
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "check-split":
        task = catalog.by_name(args.task)
        try:
            assert_trainable_split(task, args.split)
        except EvaluationSplitError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        print(json.dumps({"ok": True, "task": task.name, "split": args.split}, indent=2))
        return 0

    if args.command == "build":
        manifest = build_dataset(
            output_root=args.output,
            catalog_path=args.catalog,
            families=args.families,
            task_names=args.tasks,
            max_rows_per_split=args.max_rows_per_split,
            seed=args.seed,
            pairs_per_anchor=args.pairs_per_anchor,
            allow_cross_language_classification=not args.no_cross_language,
        )
        print(json.dumps({"build_id": manifest["build_id"], "counts": manifest["counts"]}, indent=2))
        return 0

    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
