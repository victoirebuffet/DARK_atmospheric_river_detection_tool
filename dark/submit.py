#!/usr/bin/env python3
"""Generate and optionally submit a portable DARK workflow.

The science is configured in one DARK YAML/JSON file.  This helper only
translates the same parent/children dependency graph to local, SGE, Slurm or
PBS syntax; it contains no dataset-specific settings.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import dark


def _job(path: Path, command: str, setup: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + (setup + "\n" if setup else "") + command + "\n")
    path.chmod(0o755)


def _submission_command(scheduler: str, script: Path, dependencies: list[str], options: list[str]) -> list[str]:
    if scheduler == "sge":
        return ["qsub", *options, *( ["-hold_jid", ",".join(dependencies)] if dependencies else [] ), str(script)]
    if scheduler == "slurm":
        return ["sbatch", *options, *( [f"--dependency=afterok:{':'.join(dependencies)}"] if dependencies else [] ), str(script)]
    if scheduler == "pbs":
        return ["qsub", *options, *( ["-W", f"depend=afterok:{':'.join(dependencies)}"] if dependencies else [] ), str(script)]
    raise ValueError(f"Unknown scheduler: {scheduler}")


def _job_id(scheduler: str, output: str) -> str:
    # qsub typically returns 123.server, while sbatch returns "Submitted batch
    # job 123".  Keep dots for SGE/PBS and extract the final token for Slurm.
    text = output.strip()
    return text.split()[-1] if scheduler == "slurm" else text.split()[0]


def generate(config_path: Path) -> tuple[dict, Path, dict[str, Path]]:
    cfg = dark._read_config(str(config_path))
    dark.validate_config(cfg)
    launch = cfg.get("launcher", {})
    python = launch.get("python", "python")
    setup = launch.get("environment_setup", "")
    root = Path(cfg["output_dir"]) / ".dark-submit"
    root.mkdir(parents=True, exist_ok=True)
    pipeline = Path(__file__).with_name("pipeline.py").resolve()
    quoted = lambda value: shlex.quote(str(value))
    base = f"{quoted(python)} {quoted(pipeline)}"
    config = quoted(config_path.resolve())
    scripts: dict[str, Path] = {}
    scripts["percentiles"] = root / "build-percentiles.sh"
    _job(scripts["percentiles"], f"{base} build-percentiles {config}", setup)
    for year in range(int(cfg["years"][0]), int(cfg["years"][1]) + 1):
        parent = root / f"dark-{year}.sh"; scripts[f"dark-{year}"] = parent
        _job(parent, f"{base} dark-year {config} {year}", setup)
        if cfg.get("children", {}).get("enabled"):
            child = root / f"children-{year}.sh"; scripts[f"children-{year}"] = child
            previous = "" if year == int(cfg["years"][0]) else f" --previous-parent {quoted(Path(cfg['output_dir']) / '.dark-work' / f'{year - 1}.nc')}"
            _job(child, f"{base} children-year {config} {year}{previous}", setup)
    return cfg, root, scripts


def submit(config_path: Path, scheduler: str, dry_run: bool) -> None:
    cfg, root, scripts = generate(config_path)
    options = [str(item) for item in cfg.get("launcher", {}).get("submit_options", {}).get(scheduler, [])]
    years = range(int(cfg["years"][0]), int(cfg["years"][1]) + 1)
    if scheduler == "local":
        commands = [str(scripts["percentiles"])]
        commands += [f"{shlex.quote(str(scripts[f'dark-{year}']))} &" for year in years]
        commands += ["wait"]
        if cfg.get("children", {}).get("enabled"):
            commands += [f"{shlex.quote(str(scripts[f'children-{year}']))} &" for year in years]
            commands += ["wait"]
        runner = root / "run-local.sh"; _job(runner, "\n".join(commands), "")
        print(runner)
        if not dry_run: subprocess.run([str(runner)], check=True)
        return
    commands: list[list[str]] = []
    commands.append(_submission_command(scheduler, scripts["percentiles"], [], options))
    if dry_run:
        print("\n".join(" ".join(map(shlex.quote, cmd)) for cmd in commands))
        for year in years:
            print(" ".join(map(shlex.quote, _submission_command(scheduler, scripts[f"dark-{year}"], ["PERCENTILES"], options)) ))
            if cfg.get("children", {}).get("enabled"):
                dependency = [f"DARK_{year}"] + ([] if year == int(cfg["years"][0]) else [f"DARK_{year - 1}"])
                print(" ".join(map(shlex.quote, _submission_command(scheduler, scripts[f"children-{year}"], dependency, options))))
        return
    pctl = _job_id(scheduler, subprocess.check_output(commands[0], text=True))
    parent_ids: dict[int, str] = {}
    for year in years:
        parent_ids[year] = _job_id(scheduler, subprocess.check_output(_submission_command(scheduler, scripts[f"dark-{year}"], [pctl], options), text=True))
    if cfg.get("children", {}).get("enabled"):
        for year in years:
            deps = [parent_ids[year]] + ([] if year == int(cfg["years"][0]) else [parent_ids[year - 1]])
            _job_id(scheduler, subprocess.check_output(_submission_command(scheduler, scripts[f"children-{year}"], deps, options), text=True))
    print(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/submit DARK + children workflow for local, SGE, Slurm or PBS.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--scheduler", choices=("local", "sge", "slurm", "pbs"), default="local")
    parser.add_argument("--dry-run", action="store_true", help="Write scripts and print scheduler commands without submitting.")
    args = parser.parse_args()
    submit(args.config, args.scheduler, args.dry_run)


if __name__ == "__main__": main()
