"""Run the Food.com full-catalogue recommendation experiments in dependency order."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("data", "encode", "baselines", "graph", "authors", "fusion", "evaluate", "tables")
SPLITS = ("validation", "test", "prospective")
EVALUATION_SPLITS = SPLITS[1:]


def experiment(name, *args):
    return [f"experiments/{name}.py", *args]


def commands(stage, args):
    if stage == "data":
        yield ["scripts/prepare_data.py", *(["--archive", str(args.archive)] if args.archive else [])]

    elif stage == "encode":
        yield experiment("recipe_embeddings_download")
        backend = ["--device", args.device] if args.device != "auto" else []
        yield experiment("recipe_embeddings", *backend)
        yield experiment("recipe_embeddings_examples")

    elif stage == "baselines":
        yield experiment("run")
        yield experiment("recency")
        yield experiment("ingredient_baselines")
        yield experiment("collaborative_als", "--threads", "2")

    elif stage == "graph":
        yield experiment("semantic_graph", "--split", "all")
        for split in EVALUATION_SPLITS:
            yield experiment("semantic_graph_seeds", "--split", split)

    elif stage == "authors":
        yield experiment("creator_paths", "--family", "controls")
        yield experiment("creator_paths", "--family", "graph")
        yield experiment("creator_replay")
        for split in SPLITS:
            yield experiment("author_als", "--split", split)
            yield experiment("author_als", "--split", split, "--audit")

    elif stage == "fusion":
        for split in SPLITS:
            yield experiment("graph_fusion", "--split", split)
            yield experiment("graph_fusion", "--split", split, "--verify")

    elif stage == "evaluate":
        yield experiment("semantic_graph_audit", "--split", "all")
        for split in EVALUATION_SPLITS:
            yield experiment("semantic_graph_benchmark", "--split", split)
            files = []
            for directory, suffix in (("creator_paths", "_controls"), ("creator_paths", "_graph"),
                                          ("graph_fusion", ""), ("author_als", "")):
                prefix = f"results/{directory}/{split}{suffix}"
                files.extend(["--extra", prefix + "_queries.csv", prefix + "_top10.csv"])
            yield experiment("evaluation", "--split", split, "--semantic", "--primary", "adaptive_graph_creator",
                             "--case-control", "adaptive_graph", *files)
        yield experiment("creator_replay", "--serving")
        yield experiment("creator_replay", "--distinct")
        yield experiment("figures")
        yield experiment("graph_worked_cases")

    elif stage == "tables":
        yield ["scripts/paper_tables.py", *(["--check"] if args.check else [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", metavar="STAGE", help="Default: all. Stages: " + ", ".join(STAGES))
    parser.add_argument("--encoder-python", type=Path, help="Python environment containing PyTorch and Transformers")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto", help="Backend for card encoding")
    parser.add_argument("--archive", type=Path, help="Use an existing Food.com Kaggle ZIP in the data stage")
    parser.add_argument("--plan", action="store_true", help="Print the commands without running them")
    parser.add_argument("--check", action="store_true", help="Compare the generated tables against the reference CSVs")
    args = parser.parse_args()
    invalid = set(args.stages) - set(STAGES)
    if invalid:
        parser.error("unknown stage(s): " + ", ".join(sorted(invalid)))
    if args.check and args.stages and "tables" not in args.stages:
        parser.error("--check requires the tables stage")
    if args.archive:
        args.archive = args.archive.expanduser().absolute()

    # Preserve venv symlinks: resolving a venv executable may select its base interpreter.
    default_encoder = ROOT / ".venv-encoder" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.encoder_python:
        encoder = str(args.encoder_python.expanduser().absolute())
    else:
        encoder = str(default_encoder) if default_encoder.exists() else sys.executable

    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env["PYTHONUNBUFFERED"] = "1"
    start = perf_counter()
    for stage in STAGES:
        if args.stages and stage not in args.stages:
            continue
        print(f"\n[{stage}]", flush=True)
        python = encoder if stage == "encode" else sys.executable
        for entry in commands(stage, args):
            cmd = [python, *entry]
            print(shlex.join(cmd), flush=True)
            if args.plan:
                continue
            try:
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            except FileNotFoundError as error:
                parser.exit(1, f"Cannot start {python}: {error}\n")
            except subprocess.CalledProcessError as error:
                parser.exit(error.returncode or 1, f"Failed in {stage}: {shlex.join(entry)}\n")
    if not args.plan:
        print(f"Completed in {(perf_counter() - start)/60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
