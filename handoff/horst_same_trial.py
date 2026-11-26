"""Score the impulse-only baseline using the same windows as the prescribed same-person link."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def subject_errors():
    root = Path(__file__).resolve().parents[1]
    data = root / "benchmark_outputs/cellsr1_revision/external_case_comparison_v1"
    windows = defaultdict(list)
    with (data / "gait_windows.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            windows[row["subject"]].append(row)
    joint = defaultdict(list)
    references = {}
    with (data / "gait/scored.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "one" and row["method"] == "fixed_" + row["truth"]:
                subject = row["target"]
                assert row["donor_" + row["truth"].lower()] == subject
                joint[subject, int(row["fold"])].append(float(row["pred"]))
                references[subject] = float(row["ref"])
    assert len(references) == 56 and len(joint) == 168
    person_means = []
    for subject, ref in sorted(references.items()):
        rows = windows[subject]
        trials = sorted({int(r["trial"]) for r in rows})
        errors = []
        for fold in range(3):
            arms = []
            for donor, axis in enumerate(("ap", "up")):
                ids = trials[donor::2]
                used = ids[(fold+2)%3::3] + ids[(fold+1)%3::3]
                arms.append(np.array([(float(r["xM"]), float(r["J"])) for r in rows
                                      if int(r["trial"]) in used and r["axis"] == axis]).T)
            slopes = [x @ y / (x @ x) for x, y in arms]
            weights = [1 / (len(y) * max(np.var(y), 1e-24)) for _, y in arms]
            shared = (sum(w * (x @ y) for w, (x, y) in zip(weights, arms)) /
                      sum(w * (x @ x) for w, (x, _) in zip(weights, arms)))
            assert len(joint[subject, fold]) == 55
            np.testing.assert_allclose(joint[subject, fold], shared, rtol=1e-9)
            errors.append(np.abs(np.array([*slopes, shared]) - ref))
        person_means.append(np.mean(errors, axis=0))
    return sorted(references), np.array(person_means)


def main():
    _, person_means = subject_errors()
    difference = person_means[:, 2] - person_means[:, 1]
    samples = np.random.default_rng(20261124).choice(difference, (50000, len(difference)))
    print("MAE kg (target, impulse, joint):", person_means.mean(axis=0))
    print("Joint minus impulse, mean and 95% subject bootstrap interval (kg):",
          difference.mean(), np.quantile(samples.mean(axis=1), [.025, .975]))


if __name__ == "__main__":
    main()
