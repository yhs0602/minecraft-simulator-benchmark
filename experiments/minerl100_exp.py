import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run performance comparison experiments."
    )
    # Image width options
    parser.add_argument(
        "--image_width",
        choices=["64x64", "114x64", "640x320"],
        required=True,
        help="Set the resolution of the image.",
    )

    # Load options
    parser.add_argument(
        "--load",
        choices=[
            "simulation",
            "render",
            "ppo",
            "render_ppo",
            "optimized_render",
            "optimized_ppo",
            "optimized_render_ppo",
        ],
        required=True,
        help="Specify the load configuration.",
    )

    args = parser.parse_args()

    # Display the selected configuration
    print(f"Running experiment with the following settings:")
    print(f"Image Width: {args.image_width}")
    print(f"Load: {args.load}")


if __name__ == "__main__":
    main()
