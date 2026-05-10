"""
run_all_conditions.py

Reads a counterfactual_map.json and runs eval for all three conditions:
    original      - task's default instruction
    counterfactual - distractor instruction from the map
    null          - empty string instruction

Results from each condition are saved as separate JSON files, then
a summary table is printed and saved at the end.

Usage:
    python run_all_conditions.py \
        --checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --counterfactual_map counterfactual_map.json \
        --output_dir ./results \
        --num_trials_per_task 20

    # Run only specific task IDs
    python run_all_conditions.py \
        --checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --counterfactual_map counterfactual_map.json \
        --task_ids "0,3,7" \
        --output_dir ./results \
        --num_trials_per_task 20

    # Dry run: just print what would be executed
    python run_all_conditions.py \
        --checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --counterfactual_map counterfactual_map.json \
        --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


EVAL_SCRIPT = "run_libero_eval_with_instruction.py"
CONDITIONS = ["original", "counterfactual", "null"]


def build_command(
    checkpoint,
    task_suite_name,
    instruction_override,
    task_ids_str,
    results_json,
    output_dir,
    num_trials,
    extra_args,
):
    """Build the command to run run_libero_eval_with_instruction.py."""
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--pretrained_checkpoint", checkpoint,
        "--task_suite_name", task_suite_name,
        "--instruction_override", instruction_override,
        "--results_json", results_json,
        "--num_trials_per_task", str(num_trials),
        "--local_log_dir", os.path.join(output_dir, "logs"),
    ]
    if task_ids_str:
        cmd += ["--task_ids", task_ids_str]
    if extra_args:
        cmd += extra_args.split()
    return cmd


def print_summary(all_results, output_dir, task_suite_name):
    """Print and save a comparison table across conditions."""
    print("\n" + "="*80)
    print(f"{'EXPERIMENT SUMMARY':^80}")
    print(f"Suite: {task_suite_name}")
    print("="*80)

    # Collect all task IDs
    all_task_ids = set()
    for results in all_results.values():
        all_task_ids.update(results.get("per_task", {}).keys())
    all_task_ids = sorted(all_task_ids, key=lambda x: int(x))

    # Header
    header = f"{'Task':<6} {'Instruction':<45} " + " ".join(f"{c:>16}" for c in CONDITIONS)
    print(header)
    print("-"*80)

    rows = []
    for task_id in all_task_ids:
        # Get instruction from original condition
        orig_results = all_results.get("original", {}).get("per_task", {}).get(task_id, {})
        instruction = orig_results.get("task_description", "")[:43]

        row = f"{task_id:<6} {instruction:<45} "
        for condition in CONDITIONS:
            task_data = all_results.get(condition, {}).get("per_task", {}).get(task_id)
            if task_data:
                sr = task_data["success_rate"]
                row += f"{sr:>15.1%} "
            else:
                row += f"{'N/A':>16} "
        print(row)
        rows.append(row)

    # Overall row
    print("-"*80)
    overall_row = f"{'Overall':<6} {'':<45} "
    for condition in CONDITIONS:
        sr = all_results.get(condition, {}).get("overall_success_rate")
        if sr is not None:
            overall_row += f"{sr:>15.1%} "
        else:
            overall_row += f"{'N/A':>16} "
    print(overall_row)
    print("="*80)

    # Save summary
    summary_path = os.path.join(output_dir, f"summary_{task_suite_name}.txt")
    with open(summary_path, "w") as f:
        f.write(f"EXPERIMENT SUMMARY - {task_suite_name}\n")
        f.write(f"Run at: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n")
        f.write(header + "\n")
        f.write("-"*80 + "\n")
        for row in rows:
            f.write(row + "\n")
        f.write("-"*80 + "\n")
        f.write(overall_row + "\n")
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all instruction conditions (original / counterfactual / null) using counterfactual_map.json"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--counterfactual_map", type=str, required=True,
                        help="Path to counterfactual_map.json")
    parser.add_argument("--task_ids", type=str, default="",
                        help="Comma-separated task IDs to run, e.g. '0,3,7'. Empty = run all.")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_trials_per_task", type=int, default=20)
    parser.add_argument("--extra_args", type=str, default="",
                        help="Extra arguments to pass to eval script, e.g. '--use_l1_regression True'")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--conditions", type=str, default="original,counterfactual,null",
                        help="Comma-separated conditions to run (default: all three)")
    args = parser.parse_args()

    # Load counterfactual map
    with open(args.counterfactual_map, "r") as f:
        cf_map = json.load(f)

    suite_map = cf_map.get(args.task_suite_name)
    if suite_map is None:
        print(f"[ERROR] Suite '{args.task_suite_name}' not found in {args.counterfactual_map}")
        print(f"  Available suites: {list(cf_map.keys())}")
        sys.exit(1)

    # Determine task IDs to run
    if args.task_ids.strip():
        task_ids = [x.strip() for x in args.task_ids.split(",")]
    else:
        task_ids = sorted(suite_map.keys(), key=lambda x: int(x))

    conditions_to_run = [c.strip() for c in args.conditions.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    print(f"Task suite   : {args.task_suite_name}")
    print(f"Tasks        : {task_ids}")
    print(f"Conditions   : {conditions_to_run}")
    print(f"Trials/task  : {args.num_trials_per_task}")
    print(f"Output dir   : {args.output_dir}")

    all_results = {}

    for condition in conditions_to_run:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*60}")

        results_json = os.path.join(args.output_dir, f"results_{condition}_{args.task_suite_name}.json")

        # Build per-task instruction override
        if condition == "original":
            # Run once with ORIGINAL (no per-task override needed)
            task_ids_str = ",".join(task_ids)
            cmd = build_command(
                args.checkpoint, args.task_suite_name,
                "ORIGINAL", task_ids_str,
                results_json, args.output_dir,
                args.num_trials_per_task, args.extra_args,
            )
            print(f"Command: {' '.join(cmd)}")
            if not args.dry_run:
                ret = subprocess.run(cmd)
                if ret.returncode != 0:
                    print(f"[WARNING] Condition '{condition}' exited with code {ret.returncode}")

        elif condition == "null":
            task_ids_str = ",".join(task_ids)
            cmd = build_command(
                args.checkpoint, args.task_suite_name,
                "", task_ids_str,
                results_json, args.output_dir,
                args.num_trials_per_task, args.extra_args,
            )
            print(f"Command: {' '.join(cmd)}")
            if not args.dry_run:
                ret = subprocess.run(cmd)
                if ret.returncode != 0:
                    print(f"[WARNING] Condition '{condition}' exited with code {ret.returncode}")

        elif condition == "counterfactual":
            # Each task may have a different counterfactual instruction,
            # so we run one task at a time and merge results afterward
            merged = {
                "instruction_mode": "counterfactual",
                "task_suite": args.task_suite_name,
                "total_episodes": 0,
                "total_successes": 0,
                "overall_success_rate": 0.0,
                "per_task": {},
            }

            for task_id in task_ids:
                entry = suite_map.get(str(task_id))
                if entry is None:
                    print(f"  [WARNING] Task {task_id} not found in counterfactual map, skipping.")
                    continue

                cf_instruction = entry.get("counterfactual", entry.get("original", ""))
                print(f"  Task {task_id}: \"{cf_instruction}\"")

                task_results_json = os.path.join(
                    args.output_dir, f"results_cf_task{task_id}_{args.task_suite_name}.json"
                )
                cmd = build_command(
                    args.checkpoint, args.task_suite_name,
                    cf_instruction, str(task_id),
                    task_results_json, args.output_dir,
                    args.num_trials_per_task, args.extra_args,
                )
                print(f"  Command: {' '.join(cmd)}")

                if not args.dry_run:
                    ret = subprocess.run(cmd)
                    if ret.returncode != 0:
                        print(f"  [WARNING] Task {task_id} exited with code {ret.returncode}")
                        continue

                    if os.path.exists(task_results_json):
                        with open(task_results_json, "r") as f:
                            task_result = json.load(f)
                        for tid, data in task_result.get("per_task", {}).items():
                            merged["per_task"][tid] = data
                            merged["total_episodes"] += data["episodes"]
                            merged["total_successes"] += data["successes"]

            if not args.dry_run and merged["total_episodes"] > 0:
                merged["overall_success_rate"] = merged["total_successes"] / merged["total_episodes"]
                with open(results_json, "w") as f:
                    json.dump(merged, f, indent=2)

        # Load results for summary
        if not args.dry_run and os.path.exists(results_json):
            with open(results_json, "r") as f:
                all_results[condition] = json.load(f)

    # Print summary
    if not args.dry_run and all_results:
        print_summary(all_results, args.output_dir, args.task_suite_name)
    elif args.dry_run:
        print("\n[DRY RUN] No commands were executed.")


if __name__ == "__main__":
    main()