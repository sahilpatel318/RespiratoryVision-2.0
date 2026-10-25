import argparse

from respiratory_vision.synthetic import generate_breathing_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bpm", type=float, default=18.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    meta = generate_breathing_video(
        args.output,
        bpm=args.bpm,
        duration_seconds=args.duration,
        fps=args.fps,
        seed=args.seed,
    )
    print(meta)


if __name__ == "__main__":
    main()
