import argparse
import json

from respiratory_vision.dataset import evaluate_manifest


def main():
    p = argparse.ArgumentParser(
        description="Evaluate locally authorized recordings; exported labels are deidentified."
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="outputs/real")
    p.add_argument("--kind", choices=["real", "synthetic_fixture"], default="real")
    p.add_argument("--partition", choices=["all", "development", "test"], default="all")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(
        json.dumps(
            evaluate_manifest(a.manifest, a.output, a.kind, a.partition, a.seed)[
                "methods"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
