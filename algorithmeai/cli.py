"""
CLI entry point for algorithmeai Snake classifier.
stdlib only — no external dependencies.
"""
import argparse
import sys
import json

def main():
    parser = argparse.ArgumentParser(
        prog="snake",
        description="Snake SAT-ensembled bucketed multiclass classifier v4.4.2"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    train_p = sub.add_parser("train", help="Train a Snake model from CSV")
    train_p.add_argument("csv", help="Path to CSV file")
    train_p.add_argument("--target", type=int, default=0, help="Target column index (default: 0)")
    train_p.add_argument("--layers", type=int, default=5, help="Number of layers (default: 5)")
    train_p.add_argument("--bucket", type=int, default=250, help="Bucket size (default: 250)")
    train_p.add_argument("--noise", type=float, default=0.25, help="Noise ratio (default: 0.25)")
    train_p.add_argument("--output", "-o", default="snakeclassifier.json", help="Output JSON path")
    train_p.add_argument("--vocal", action="store_true", help="Verbose output")

    # predict
    pred_p = sub.add_parser("predict", help="Predict using a saved model")
    pred_p.add_argument("model", help="Path to saved JSON model")
    pred_p.add_argument("--query", "-q", required=True, help="JSON dict to classify")
    pred_p.add_argument("--audit", action="store_true", help="Show full audit")

    # info
    info_p = sub.add_parser("info", help="Show model info")
    info_p.add_argument("model", help="Path to saved JSON model")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from .snake import Snake

    if args.command == "train":
        s = Snake(
            args.csv,
            target_index=args.target,
            n_layers=args.layers,
            bucket=args.bucket,
            noise=args.noise,
            vocal=args.vocal,
            saved=True
        )
        s.to_json(args.output)
        print(f"Model saved to {args.output}")
        print(f"Population: {len(s.population)}, Layers: {len(s.layers)}")

    elif args.command == "predict":
        s = Snake(args.model)
        X = json.loads(args.query)
        pred = s.get_prediction(X)
        prob = s.get_probability(X)
        print(f"Prediction: {pred}")
        print(f"Confidence: {100*max(prob.values()):.1f}%")
        if args.audit:
            print(s.get_audit(X))

    elif args.command == "info":
        with open(args.model) as f:
            data = json.load(f)
        version = data.get("version", "0.1.0")
        n_pop = len(data.get("population", []))
        n_layers = len(data.get("layers", []))
        target = data.get("target", "?")
        print(f"Snake model v{version}")
        print(f"Target: {target}")
        print(f"Population: {n_pop}")
        print(f"Layers: {n_layers}")


if __name__ == "__main__":
    main()
