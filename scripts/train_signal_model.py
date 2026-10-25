import argparse
import json

from respiratory_vision.training import TrainingConfig, train_model


def main():
    p = argparse.ArgumentParser(
        description="Optional deterministic CPU CNN experiment on synthetic signals"
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--train-size", type=int, default=600)
    p.add_argument("--validation-size", type=int, default=120)
    p.add_argument("--test-size", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="outputs/model/temporal_cnn.pt")
    a = p.parse_args()
    output = a.output
    delattr(a, "output")
    result = train_model(TrainingConfig(**vars(a)), output)
    print(
        json.dumps(
            {
                "best_epoch": result["best_epoch"],
                "test_comparison": result["test_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
