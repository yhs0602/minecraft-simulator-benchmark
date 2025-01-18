import wandb


if __name__ == "__main__":
    api = wandb.Api()
    r = api.run("jourhyang123/minecraft-envs-performance-comparison/2fvxiees")
    print(f"Updating {r.name}")
    r.group = "cpu-craftground-apple-raw--64-64-sbx-ppo"
    r.update()
