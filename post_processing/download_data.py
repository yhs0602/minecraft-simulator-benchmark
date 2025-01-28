import os
from typing import Dict

import pandas as pd
import tqdm
import wandb
from wandb.apis.public import Run, Runs

current_dir = os.path.dirname(os.path.realpath(__file__))


# Axis: x: wall clock time, y: global step
# Axis: x: global step, y: time/fps
# target: top 4 runs in terms of final time/fps
def download_data(run: Run, download_dir) -> Dict:
    run_id = run.id
    run_group = run.group
    group_dir = os.path.join(download_dir, run_group)
    os.makedirs(group_dir, exist_ok=True)
    file_path = os.path.join(group_dir, f"{run_id}.csv")
    print(f"Downloading {run_id} to {file_path}")
    data = {
        "group": run.group,
        "id": run.id,
        "runtime": run.summary.get("_wandb.runtime"),
        "global_step": run.summary.get("global_step")
        or run.summary.get("time/iterations"),
        "time/fps": run.summary.get("time/fps"),
    }
    print(f"Data of {run.name} ({run.group}): {data}")

    # print(f"Summary: {run.summary}")
    return data
    if not os.path.exists(file_path):
        history = run.scan_history()
        columns = [
            "step",
            "global_step",
        ]
        data = [[row.get(column) for column in columns] for row in history]
        df = pd.DataFrame(columns=columns, data=data)
        df.to_csv(file_path)
    else:
        print(f"File already exists, skipping {file_path}")
        return


def predicate(run: Run):
    # v2
    # no cpu
    # no apple
    group_names = [
        "v2-minerl100--64-64-render",
        "v2-minerl100--64-64-render_ppo",
        "v2-minerl100--64-64-render_sbx-ppo",
        "v2-minerl100--640-360-render",
        "v2-minerl100--640-360-render_ppo",
        "v2-minerl100--640-360-render_sbx-ppo",
        "v2-craftground-raw--64-64-render",
        "v2-craftground-raw--64-64-render_ppo",
        "v2-craftground-raw--64-64-render_sbx-ppo",
        "v2-craftground-raw--640-360-render",
        "v2-craftground-raw--640-360-render_ppo",
        "v2-craftground-raw--640-360-render_sbx-ppo",
    ]
    return run.group in group_names


if __name__ == "__main__":
    api = wandb.Api(timeout=120)
    WANDB_ENTITY = "jourhyang123"
    WANDB_PROJECT = "minecraft-envs-performance-comparison"
    runs: Runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    group_runs = [run for run in runs if predicate(run)]
    print(f"Found {len(group_runs)} runs")
    out_dir = os.path.join(current_dir, "data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    all_data = []
    for run in tqdm.tqdm(group_runs):
        one_data = download_data(run, os.path.join(out_dir, out_dir))
        all_data.append(one_data)

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(out_dir, "all.csv"))
