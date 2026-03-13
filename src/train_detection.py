import argparse

from train import run_training


DEFAULT_CONFIG_PATH = "configs/detection.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an RF-DETR detection model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the detection training config YAML file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
