import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def group_fps_top4(csv_file: str) -> dict:
    topk = 1
    df = pd.read_csv(csv_file)

    # Seperate 64x64 640x360
    df_64x64 = df[df["group"].str.contains("64-64")]
    df_640x360 = df[df["group"].str.contains("640-360")]

    # Select top topk runs for each group
    top_64x64 = (
        df_64x64.groupby("group")
        .apply(lambda x: x.nlargest(topk, "time/fps"))
        .reset_index(drop=True)
    )
    top_640x360 = (
        df_640x360.groupby("group")
        .apply(lambda x: x.nlargest(topk, "time/fps"))
        .reset_index(drop=True)
    )

    new_labels = {
        "v2-craftground-raw--64-64-render_sbx-ppo": "CraftGround SBX",
        "v2-craftground-raw--64-64-render_ppo": "CraftGround SB3",
        "v2-craftground-raw--64-64-render": "CraftGround",
        "v2-minerl100--64-64-render_sbx-ppo": "Malmo SBX",
        "v2-minerl100--64-64-render_ppo": "Malmo SB3",
        "v2-minerl100--64-64-render": "Malmo",
        "v2-craftground-raw--640-360-render_sbx-ppo": "CraftGround SBX",
        "v2-craftground-raw--640-360-render_ppo": "CraftGround SB3",
        "v2-craftground-raw--640-360-render": "CraftGround",
        "v2-minerl100--640-360-render_sbx-ppo": "Malmo SBX",
        "v2-minerl100--640-360-render_ppo": "Malmo SB3",
        "v2-minerl100--640-360-render": "Malmo",
    }

    # 64x64 Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=top_64x64, x="group", y="time/fps", palette="Set3", width=0.5)
    sns.swarmplot(data=top_64x64, x="group", y="time/fps", color=".25")
    plt.title(f"Top {topk} Final FPS Comparison (64x64)", fontsize=16)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Final FPS", fontsize=12)
    current_labels = top_64x64["group"].unique()  # Get current group names
    plt.xticks(
        ticks=range(len(current_labels)),  # Number of indices = number of groups
        labels=[
            new_labels.get(label, label) for label in current_labels
        ],  # Apply new labels
        rotation=0,
        ha="center",
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    y_min, y_max = plt.ylim()  # Get current y-axis range
    offset = (y_max - y_min) * 0.04
    for i, group in enumerate(top_64x64["group"].unique()):
        avg_fps = top_64x64[top_64x64["group"] == group]["time/fps"].mean()
        top_fps = top_64x64[top_64x64["group"] == group]["time/fps"].max()
        min_fps = top_64x64[top_64x64["group"] == group]["time/fps"].min()
        plt.text(
            i,
            min_fps - offset,
            f"{avg_fps:.1f}",
            ha="center",
            fontsize=10,
            color="blue",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.tight_layout()
    plt.margins(y=0.1)
    plt.savefig(f"64x64-{topk}.png")
    plt.show()

    # 640x360 Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=top_640x360, x="group", y="time/fps", palette="Set3", width=0.5)
    sns.swarmplot(data=top_640x360, x="group", y="time/fps", color=".25")
    plt.title(f"Top {topk} Final FPS Comparison (640x360)", fontsize=16)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Final FPS", fontsize=12)
    current_labels = top_640x360["group"].unique()  # Get current group names
    plt.xticks(
        ticks=range(len(current_labels)),  # Number of indices = number of groups
        labels=[
            new_labels.get(label, label) for label in current_labels
        ],  # Apply new labels
        rotation=0,
        ha="center",
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    y_min, y_max = plt.ylim()  # Get current y-axis range
    offset = (y_max - y_min) * 0.04
    for i, group in enumerate(top_640x360["group"].unique()):
        avg_fps = top_640x360[top_640x360["group"] == group]["time/fps"].mean()
        top_fps = top_640x360[top_640x360["group"] == group]["time/fps"].max()
        min_fps = top_640x360[top_640x360["group"] == group]["time/fps"].min()
        plt.text(
            i,
            min_fps - offset,
            f"{avg_fps:.1f}",
            ha="center",
            fontsize=10,
            color="blue",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
    plt.tight_layout()
    plt.margins(y=0.1)
    plt.savefig(f"640x360-{topk}.png")
    plt.show()

    # Return grouped dictionary
    grouped_dict = {"64x64_topk": top_64x64, "640x360_topk": top_640x360}
    return grouped_dict


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "data", "all.csv")
    grouped_dict = group_fps_top4(csv_file)
    print(grouped_dict)
