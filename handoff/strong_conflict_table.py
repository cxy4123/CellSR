"""Rebuild Table 1b from matched strong-conflict estimates (no new fits)."""
import csv
from pathlib import Path

import numpy as np


def main():
    base = Path(__file__).resolve().parents[1] / "benchmark_outputs/cellsr1_revision"
    with (base / "search_path_classsr_v1/ablation.csv").open(newline="") as f:
        independent = {(int(r["seed"]), r["family"]): float(r["theta"]) for r in csv.DictReader(f)
                       if (r["environment"], r["link"], r["method"]) == ("heterogeneous", "conflict", "A")}
    with (base / "soft_decisions/cases.csv").open(newline="") as f:
        paired = {(int(r["seed"]), r["family"]): r for r in csv.DictReader(f) if float(r["theta"]) == 2.1}
    keys = sorted(independent)
    assert keys == sorted(paired) and len(keys) == 128
    a = np.array([independent[k] for k in keys])
    b = np.array([float(paired[k]["a"]) for k in keys])
    for label, estimates in (("independent", a), ("soft", b), ("CellSR", a)):
        np.testing.assert_allclose(abs(estimates-1.2), [float(paired[k][label]) for k in keys], atol=1e-10, rtol=0)
    assert all(r["accepted"] == "False" for r in paired.values())
    with (base / "soft_decisions/strong_conflict_table.csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method","mean_signed_error","across_task_sd_ddof0","mae"])
        for label, estimates in (("independent", a), ("soft", b)):
            writer.writerow([label,(estimates-1.2).mean(), estimates.std(ddof=0),abs(estimates-1.2).mean()])
    seeds = np.array([k[0] for k in keys])
    unique, counts = np.unique(seeds, return_counts=True)
    assert len(unique)==64 and np.all(counts==2)
    diff = abs(b-1.2)-abs(a-1.2)
    by_seed = [diff[seeds==s].mean() for s in unique]
    samples = np.random.default_rng(20261124).choice(by_seed,(20000,len(unique)))
    print("Soft minus independent MAE:",diff.mean(),np.quantile(samples.mean(axis=1),[.025,.975]))
    print("Unchanged/up/down:",sum(abs(b-a)<=1e-10),sum(b-a>1e-10),sum(b-a < -1e-10))


if __name__=="__main__":
    main()
