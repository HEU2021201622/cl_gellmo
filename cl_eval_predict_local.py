import argparse
from pathlib import Path

from cl_config import load_cl_eval_config
from cl_eval_data import collect_prediction_candidates, discover_jobs
from cl_eval_props import predict_local_properties_for_smiles
from cl_manifest import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Predict local evaluation properties for CL outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["sequence", "joint"], default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--joint-name", default=None)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--steps", nargs="+", default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--cache-path", default=None)
    return parser.parse_args()


def apply_overrides(config, args):
    cache_overridden = args.cache_path is not None
    if args.mode:
        config["mode"] = args.mode
    if args.method:
        config["method"] = args.method
    if args.joint_name:
        config["joint_name"] = args.joint_name
    if args.input_root:
        config["input_root"] = args.input_root
    if args.train_root:
        config["train_root"] = args.train_root
    if args.output_root:
        config["output_root"] = args.output_root
    if args.settings:
        config["settings"] = args.settings
    if args.steps:
        config["_steps_filter"] = set(args.steps)
    if args.runs:
        config["_runs_filter"] = set(args.runs)
    if args.tasks:
        config["_tasks_filter"] = set(args.tasks)
    if args.cache_path:
        config["local_cache_path"] = args.cache_path
    if args.output_root and not cache_overridden:
        config["local_cache_path"] = str(Path(config["output_root"]) / "property_cache_local.csv")
    return config


def main():
    args = parse_args()
    config = apply_overrides(load_cl_eval_config(args.config), args)
    jobs = discover_jobs(config)
    smiles, properties = collect_prediction_candidates(jobs)
    local_properties = [prop for prop in properties if prop in {"drd2", "qed", "plogp"}]
    ensure_dir(str(Path(config["local_cache_path"]).parent))
    df = predict_local_properties_for_smiles(
        smiles,
        local_properties + ["sas"],
        cache_path=config["local_cache_path"],
    )
    print(f"[eval-local] wrote {len(df)} rows to {config['local_cache_path']}")


if __name__ == "__main__":
    main()
