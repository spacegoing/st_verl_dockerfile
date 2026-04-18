# W&B Offline Run Extraction Script
"""
W&B Offline Run Extraction Script

Reads .wandb binary protobuf files and extracts:
- actor/* metrics (sampled every TRAIN_SAMPLE_INTERVAL steps)
- val-core/* metrics (all, typically every 10 steps)

Safety: copies wandb dir to /tmp (local disk) before processing.
CSVs written to persistent PFS location.
"""

import json
import os
import glob
import csv
import subprocess
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

# ─── Config ───
WANDB_DIR = "/root/myCodeLab/host/verl/wandb_my_dirs/wandb"
EXTRACT_BASE = "/root/myCodeLab/host/verl/wandb_extract"
os.makedirs(EXTRACT_BASE, exist_ok=True)
PUBLISH_DIR = "/mnt/llm-train/users/lichang93/wandb_stats_tables"
TRAIN_SAMPLE_INTERVAL = 5
FLOAT_PRECISION = 8


def setup_work_dir(run_name):
    """
    Copy wandb source to /tmp (local disk, avoids PFS locking).
    CSVs go to EXTRACT_BASE on PFS (persistent).

    Returns:
        (copy_dir, output_dir)
    """
    copy_dir = f"/tmp/wandb_extract_{run_name}"
    output_dir = os.path.join(EXTRACT_BASE, run_name, "csvs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output dir: {output_dir}")
    print(f"Copying {WANDB_DIR} -> {copy_dir} (local /tmp) ...")
    subprocess.run(["cp", "-r", WANDB_DIR, copy_dir], check=True)
    print("Copy done.\n")

    return copy_dir, output_dir


def parse_run(wandb_file_path):
    """
    Parse a .wandb binary file and extract run_name + history rows.

    Returns:
        run_name: str or None
        history_rows: list of dict, each dict is one wandb.log() call
    """
    file_size = os.path.getsize(wandb_file_path)
    print(f"  parsing {os.path.basename(wandb_file_path)} ({file_size / 1024:.1f} KB) ...")

    ds = DataStore()
    ds.open_for_scan(wandb_file_path)

    run_name = None
    history_rows = []
    record_count = 0

    while True:
        data = ds.scan_data()
        if data is None:
            break
        record_count += 1
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue

        if rec.HasField("run") and rec.run.display_name:
            run_name = rec.run.display_name

        if rec.HasField("history"):
            row = parse_history_record(rec.history)
            if row:
                history_rows.append(row)

    print(f"  parsed {record_count} records, {len(history_rows)} history rows")
    return run_name, history_rows


def verify_completeness(history_rows, run_dir):
    """
    Verify extracted history against wandb-summary.json.

    Checks:
    1. Max _step in history vs _step in summary
    2. Whether all expected steps are present (no gaps)

    Returns True if verification passes.
    """
    summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
    if not os.path.exists(summary_path):
        print("  WARN: no wandb-summary.json to verify against")
        return True

    with open(summary_path) as f:
        summary = json.load(f)

    summary_step = summary.get("_step")
    if summary_step is None:
        print("  WARN: no _step in wandb-summary.json")
        return True

    steps = sorted(set(int(row["_step"]) for row in history_rows if "_step" in row))

    if not steps:
        print("  FAIL: no steps found in history")
        return False

    history_max = steps[-1]
    ok = True

    if history_max < summary_step:
        print(f"  WARN: history max step ({history_max}) < summary step ({summary_step})")
        print(f"        missing {summary_step - history_max} steps at the end")
        ok = False
    elif history_max == summary_step:
        print(f"  OK: history max step matches summary ({summary_step})")

    expected_steps = set(range(0, history_max + 1))
    missing = expected_steps - set(steps)
    if missing:
        print(f"  WARN: {len(missing)} missing steps in 0..{history_max}: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")
        ok = False
    else:
        print(f"  OK: all {len(steps)} steps present (0..{history_max}), no gaps")

    return ok


def parse_history_record(history):
    """
    Parse a single history record into a dict of scalar metrics.

    Note: wandb uses `nested_key` (a repeated/list field) not `key`
    for history items. The key name is in nested_key[0].
    """
    row = {}
    for item in history.item:
        k = item.nested_key[0] if len(item.nested_key) > 0 else item.key
        if not k:
            continue
        try:
            val = json.loads(item.value_json)
            if isinstance(val, (int, float)):
                row[k] = val
        except Exception:
            pass
    return row


def filter_keys(history_rows):
    """
    Collect all keys across history rows and return filtered subsets.

    Returns:
        actor_keys: sorted list of keys starting with "actor"
        valcore_keys: sorted list of keys starting with "val-core"
    """
    all_keys = set(k for row in history_rows for k in row.keys())
    actor_keys = sorted(k for k in all_keys if k.startswith("actor"))
    valcore_keys = sorted(k for k in all_keys if k.startswith("val-core"))
    return actor_keys, valcore_keys


def group_by_step(history_rows):
    """
    Merge all history rows by _step.

    Multiple wandb.log() calls on the same step get merged into one dict.
    """
    step_data = {}
    for row in history_rows:
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        if step not in step_data:
            step_data[step] = {}
        step_data[step].update(row)
    return step_data


def round_value(val):
    """Round floats to FLOAT_PRECISION digits. Pass through ints."""
    if isinstance(val, float):
        return round(val, FLOAT_PRECISION)
    return val


def build_output_rows(step_data, actor_keys, valcore_keys):
    """
    Build downsampled output rows.

    - Actor columns populated every TRAIN_SAMPLE_INTERVAL steps, else None.
    - Val-core columns populated only on steps with validation data, else None.
    - Steps matching neither condition are dropped.
    """
    output_rows = []
    for step in sorted(step_data.keys()):
        data = step_data[step]
        has_val = any(k in data for k in valcore_keys)
        on_train_boundary = (step % TRAIN_SAMPLE_INTERVAL == 0)

        if not has_val and not on_train_boundary:
            continue

        out = {"_step": step}

        for k in actor_keys:
            out[k] = round_value(data[k]) if (on_train_boundary and k in data) else None

        for k in valcore_keys:
            out[k] = round_value(data[k]) if (has_val and k in data) else None

        output_rows.append(out)

    return output_rows


def write_csv(output_rows, ordered_keys, csv_path):
    """Write rows to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(output_rows)


def process_run(run_dir, output_dir):
    """
    Process a single offline run directory.

    Returns the csv path written, or None if skipped.
    """
    run_id = os.path.basename(run_dir).split("-")[-1]
    print(f"Processing {run_id} ({os.path.basename(run_dir)}) ...")
    wandb_files = [f for f in os.listdir(run_dir) if f.endswith(".wandb")]
    if not wandb_files:
        print(f"SKIP {run_id}: no .wandb file")
        return None

    # Parse
    run_name, history_rows = parse_run(os.path.join(run_dir, wandb_files[0]))
    run_name = run_name or run_id

    print(f"{run_id} ({run_name}): {len(history_rows)} history rows")
    if not history_rows:
        return None

    # Verify
    verify_completeness(history_rows, run_dir)

    # Filter & group
    actor_keys, valcore_keys = filter_keys(history_rows)
    print(f"  actor keys ({len(actor_keys)}): {actor_keys}")
    print(f"  val-core keys ({len(valcore_keys)}): {valcore_keys}")

    step_data = group_by_step(history_rows)
    output_rows = build_output_rows(step_data, actor_keys, valcore_keys)

    # Write
    ordered_keys = ["_step"] + actor_keys + valcore_keys
    safe_name = run_name.replace("/", "_").replace(" ", "_")
    csv_path = os.path.join(output_dir, f"{safe_name}.csv")
    write_csv(output_rows, ordered_keys, csv_path)

    print(f"  -> {csv_path} ({len(output_rows)} rows)\n")
    return csv_path


def main():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    # Step 1: Copy to local /tmp
    copy_dir, output_dir = setup_work_dir(run_name)

    # Step 2: Process each run
    run_dirs = sorted(glob.glob(os.path.join(copy_dir, "offline-run-*")))
    print(f"Found {len(run_dirs)} offline runs.\n")

    for run_dir in run_dirs:
        process_run(run_dir, output_dir)

    # Step 3: Cleanup /tmp copy (local disk, rm -rf always works)
    print(f"Cleaning up {copy_dir} ...")
    subprocess.run(["rm", "-rf", copy_dir])

    # Step 4: Compress and publish
    run_dir_path = os.path.dirname(output_dir)  # .../wandb_extract/run_XXXXXX/
    tar_name = f"{run_name}.tar.gz"
    tar_path = os.path.join(EXTRACT_BASE, tar_name)

    print(f"Compressing {run_dir_path} -> {tar_path} ...")
    subprocess.run(
        ["tar", "-czf", tar_path, "-C", EXTRACT_BASE, run_name],
        check=True,
    )

    os.makedirs(PUBLISH_DIR, exist_ok=True)
    subprocess.run(["chmod", "755", PUBLISH_DIR], check=True)
    publish_path = os.path.join(PUBLISH_DIR, tar_name)
    subprocess.run(["mv", tar_path, publish_path], check=True)
    subprocess.run(["chmod", "644", publish_path], check=True)
    print(f"Published: {publish_path}")

    print(f"\nDone. CSVs also in: {output_dir}")


if __name__ == "__main__":
    main()
(base) root@daxingG5-jump:~/myCodeLab/host/verl/wandb_my_dirs#
