import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def select_middlek(df, k):
    sorted_df = df.sort_values("time/fps", ascending=False)
    mid_idx = len(sorted_df) // 2
    half_k = k // 2
    return sorted_df.iloc[
        max(0, mid_idx - half_k) : min(len(sorted_df), mid_idx + half_k + k % 2)
    ]


def select_topk(df, k):
    return df.nlargest(k, "time/fps")


def remove_outliers_iqr(df, column="time/fps"):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


desired_order = [
    "CraftGround",
    "CraftGround SB3",
    "CraftGround SBX",
    "Malmo",
    "Malmo SB3",
    "Malmo SBX",
]


def group_fps_topk(
    csv_file: str, k: int, use_middle: bool = True, remove_iql: bool = True
) -> dict:
    df = pd.read_csv(csv_file)

    desired_order = [
        "CraftGround",
        "CraftGround SB3",
        "CraftGround SBX",
        "Malmo",
        "Malmo SB3",
        "Malmo SBX",
    ]

    new_labels = {
        "v3-craftground-raw--64-64-render_sbx-ppo": "CraftGround SBX",
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

    df_64x64 = df[df["group"].str.contains("64-64")].copy()
    df_640x360 = df[df["group"].str.contains("640-360")].copy()

    df_64x64["group"] = df_64x64["group"].map(new_labels)
    df_64x64 = df_64x64.sort_values(
        by="group",
        key=lambda x: x.map({v: i for i, v in enumerate(desired_order)}),
    )

    df_640x360["group"] = df_640x360["group"].map(new_labels)
    df_640x360 = df_640x360.sort_values(
        by="group",
        key=lambda x: x.map({v: i for i, v in enumerate(desired_order)}),
    )

    if remove_iql:
        df_64x64 = df_64x64.groupby("group", group_keys=False).apply(
            remove_outliers_iqr
        )
        df_640x360 = df_640x360.groupby("group", group_keys=False).apply(
            remove_outliers_iqr
        )

    if use_middle:
        # Select middle topk runs for each group
        selected_64x64 = (
            df_64x64.groupby("group")
            .apply(lambda x: select_middlek(x, k))
            .reset_index(drop=True)
        )
        selected_640x360 = (
            df_640x360.groupby("group")
            .apply(lambda x: select_middlek(x, k))
            .reset_index(drop=True)
        )
    else:
        # Select top topk runs for each group
        selected_64x64 = (
            df_64x64.groupby("group")
            .apply(lambda x: x.nlargest(k, "time/fps"))
            .reset_index(drop=True)
        )
        selected_640x360 = (
            df_640x360.groupby("group")
            .apply(lambda x: x.nlargest(k, "time/fps"))
            .reset_index(drop=True)
        )

    # 64x64 Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=selected_64x64,
        x="group",
        y="time/fps",
        palette="Set3",
        width=0.5,
    )
    sns.swarmplot(
        data=selected_64x64,
        x="group",
        y="time/fps",
        color=".25",
    )
    title_text = "Middle" if use_middle else "Top"
    plt.title(f"{title_text} {k} Final FPS Comparison (64x64)", fontsize=16)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Final FPS", fontsize=12)
    current_labels = selected_64x64["group"].unique()  # Get current group names
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
    for i, group in enumerate(selected_64x64["group"].unique()):
        avg_fps = selected_64x64[selected_64x64["group"] == group]["time/fps"].mean()
        top_fps = selected_64x64[selected_64x64["group"] == group]["time/fps"].max()
        min_fps = selected_64x64[selected_64x64["group"] == group]["time/fps"].min()
        std_fps = selected_64x64[selected_64x64["group"] == group]["time/fps"].std()
        plt.text(
            i,
            min_fps - offset,
            f"{avg_fps:.1f} ({std_fps:.1f})",
            ha="center",
            fontsize=10,
            color="blue",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.tight_layout()
    plt.margins(y=0.1)
    plt.savefig(f"images/64x64-{title_text.lower()}{k}.png")
    plt.show()

    # 640x360 Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=selected_640x360,
        x="group",
        y="time/fps",
        palette="Set3",
        width=0.5,
    )
    sns.swarmplot(
        data=selected_640x360,
        x="group",
        y="time/fps",
        color=".25",
    )
    plt.title(f"{title_text} {k} Final FPS Comparison (640x360)", fontsize=16)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Final FPS", fontsize=12)
    current_labels = selected_640x360["group"].unique()  # Get current group names
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
    for i, group in enumerate(selected_640x360["group"].unique()):
        avg_fps = selected_640x360[selected_640x360["group"] == group][
            "time/fps"
        ].mean()
        top_fps = selected_640x360[selected_640x360["group"] == group]["time/fps"].max()
        min_fps = selected_640x360[selected_640x360["group"] == group]["time/fps"].min()
        std_fps = selected_640x360[selected_640x360["group"] == group]["time/fps"].std()
        plt.text(
            i,
            min_fps - offset,
            f"{avg_fps:.1f} ({std_fps:.1f})",
            ha="center",
            fontsize=10,
            color="blue",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
    plt.tight_layout()
    plt.margins(y=0.1)
    plt.savefig(f"images/640x360-{title_text.lower()}{k}.png")
    plt.show()

    # Return grouped dictionary
    grouped_dict = {"64x64_topk": selected_64x64, "640x360_topk": selected_640x360}
    return grouped_dict


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "data", "all.csv")
    grouped_dict = group_fps_topk(csv_file, k=9)
    print(grouped_dict)
