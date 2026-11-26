"""Conditional target-subject bootstrap for the revised manuscript, section 4.3."""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    data = Path(__file__).resolve().parents[1] / "benchmark_outputs/cellsr1_revision/matched_false_share"
    sealed = data / "sealed_predictions.csv"
    assert hashlib.sha256(sealed.read_bytes()).hexdigest() == (data / "sealed_predictions.sha256").read_text().strip()
    with (data / "publisher_mass.csv").open(newline="") as f:
        mass = {r["subject"]: float(r["mass"]) for r in csv.DictReader(f)}
    errors = defaultdict(list)
    with sealed.open(newline="") as f:
        for row in csv.DictReader(f):
            target = row["target"]
            predictions = [float(row["m-"]), float(row["m" + row["B"]])]
            errors[target, row["kind"]].append(np.abs(np.array(predictions) - mass[target]))
    assert len(mass) == 22 and len(errors) == 44
    assert all(len(group) == 42 for group in errors.values())
    subject_errors = np.array([
        np.mean([np.mean(errors[s, k], axis=0) for k in ("one", "zero")], axis=0)
        for s in sorted(mass)
    ])
    gains = subject_errors[:, 0] - subject_errors[:, 1]
    samples = np.random.default_rng(20261124).choice(gains, (50000, len(gains)))
    interval = np.quantile(samples.mean(axis=1), [.025, .975])
    print("MAE (independent, CellSR):", subject_errors.mean(axis=0))
    print("Paired improvement (kg):", gains.mean())
    print("95% percentile interval, fixed candidates:", interval)


if __name__ == "__main__":
    main()
