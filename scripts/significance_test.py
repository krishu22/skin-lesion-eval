import argparse
import json
import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired significance test between two configs' per-fold test balanced accuracy."
    )
    parser.add_argument("--file", type=str, default=None, help='JSON file: {"config_a": [...], "config_b": [...]}')
    parser.add_argument("--a", type=float, nargs=5, default=None, help="Config A's 5 per-fold values")
    parser.add_argument("--b", type=float, nargs=5, default=None, help="Config B's 5 per-fold values")
    parser.add_argument("--name-a", type=str, default="Config A")
    parser.add_argument("--name-b", type=str, default="Config B")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
        a, b = data["config_a"], data["config_b"]
    elif args.a is not None and args.b is not None:
        a, b = args.a, args.b
    else:
        raise ValueError("Provide either --file or both --a and --b")

    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired test requires equal-length lists, got {len(a)} and {len(b)}")

    t_stat, t_p = stats.ttest_rel(a, b)
    w_stat, w_p = stats.wilcoxon(a, b)

    print(f"{args.name_a}: mean={a.mean():.4f} +/- std={a.std(ddof=1):.4f} ({[float(x) for x in a]})")
    print(f"{args.name_b}: mean={b.mean():.4f} +/- std={b.std(ddof=1):.4f} ({[float(x) for x in b]})")
    print(f"\nPaired t-test:          t={t_stat:.4f}, p={t_p:.4f}")
    print(f"Wilcoxon signed-rank:   W={w_stat:.4f}, p={w_p:.4f}")


if __name__ == "__main__":
    main()
