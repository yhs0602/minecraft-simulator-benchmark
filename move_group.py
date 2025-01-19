import wandb

# filter out
# No jax-metal
# No proper cpu setting in cpu
if __name__ == "__main__":
    api = wandb.Api()
    run_ids = {
        "unkcpu-craftground-apple-raw--640-360-render_sbx-ppo": [
            "mdgqwa9e",
            "rm7mhgli",
        ],
        "unkcpu-craftground-apple-raw--64-64-render_sbx-ppo": [
            "brg3slmf",
        ],
        "unkcpu-craftground-apple-raw--64-64-sbx-ppo": [
            "7tfj073b",
            "eaxpeh6z",
            "qi18c19k",
            "2fvxiees",
        ],
        "unkcpu-craftground-apple-raw--640-360-sbx-ppo": [
            "qyv3m27c",
            "kr4obhod",
            "kr4obhod",
        ],
        "unk-craftground-apple-raw--64-64-sbx-ppo": [
            "7lmf22f1",
            "42sndnla",
        ],
        "unk-craftground-apple-raw--640-360-sbx-ppo": [
            "kv8d1s2j",
        ],
    }
    for k, v in run_ids.items():
        for run_id in v:
            r = api.run(f"jourhyang123/minecraft-envs-performance-comparison/{run_id}")
            print(f"Updating {r.name}")
            r.group = k
            r.update()
