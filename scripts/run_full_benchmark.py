import argparse
import json

from respiratory_vision.benchmarking import run_benchmark


def main():
    p = argparse.ArgumentParser(
        description="Reproducible synthetic engineering benchmark"
    )
    p.add_argument("--output", default="outputs/benchmark")
    p.add_argument("--split", choices=["development", "test"], default="test")
    p.add_argument("--per-condition", type=int, default=4)
    p.add_argument("--checkpoint")
    a = p.parse_args()
    summary = run_benchmark(a.output, a.split, a.per_condition, checkpoint=a.checkpoint)
    from respiratory_vision.reporting import write_benchmark_report

    write_benchmark_report(a.output)
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
