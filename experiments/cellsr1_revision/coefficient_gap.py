"""Table 1 sensitivity analysis with matched noise and frozen local formulas."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from search_path_ablation import ALL, OUT, ROOT, dataset, label, law
from cellsr.joint_information import declared_star_links, fit_linear_links, select_declared_links, validation_losses


def soft_fit(laws, links, independent, joint):
    candidates = [independent] + [fit_linear_links(laws, links, penalty=p)
                                  for p in (.001, .01, .1, 1, 10, 100, 1000)] + [joint]
    return min(candidates, key=lambda fit: validation_losses(laws, fit).mean())


def main():
    destination = OUT.parent / "coefficient_gap"
    destination.mkdir(exist_ok=True)
    archive = pd.read_csv(OUT / "ablation.csv").query("environment=='heterogeneous' and method=='A'")
    keys = ["family", "seed"]
    assert archive.groupby(keys)[["formula", "donor_formula"]].nunique().eq(1).all().all()
    paths = {label(path): path for path in ALL}
    rows = []
    for case in archive.query("link=='valid'").itertuples():
        target, donor = dataset(case.seed, case.family, "heterogeneous", "valid")
        target_law = law("target", target, paths[case.formula], 2)
        for theta in (1.2, 1.3, 1.4, 1.6, 1.8, 2.1):
            shifted = [(x, z, y + (theta - 1.2) * x) for x, z, y in donor]
            laws = [target_law, law("donor", shifted, paths[case.donor_formula], 2)]
            links = declared_star_links(laws)
            independent = fit_linear_links(laws)
            joint = fit_linear_links(laws, links)
            soft = soft_fit(laws, links, independent, joint)
            selected = select_declared_links(laws)
            methods = dict(independent=independent, joint=joint, soft=soft, CellSR=selected.estimate)
            errors = {name: abs(est.coefficients["target"][1] - 1.2) for name, est in methods.items()}
            rows.append(dict(seed=case.seed, family=case.family, theta=theta,
                             accepted=bool(selected.links), **errors))
    frame = pd.DataFrame(rows)
    metrics = list(methods)
    original = pd.read_csv(ROOT / "papers/CellSR_MLST_Chinese/revision_zh/simulation_table.csv")
    for theta, link in [(1.2, "valid"), (2.1, "conflict")]:
        old = original.query("link==@link").set_index(keys).sort_index()
        new = frame.query("theta==@theta").set_index(keys).sort_index()
        assert new.index.equals(old.index) and new.accepted.equals(old.accepted)
        np.testing.assert_allclose(new[metrics], old[metrics], atol=1e-10, rtol=0)
    assert frame.groupby("theta").size().eq(128).all()
    frame.to_csv(destination / "cases.csv", index=False)
    summary = frame.groupby("theta")[metrics].mean()
    summary["accepted"] = frame.groupby("theta").accepted.sum()
    # Preserve both families within each seed and pair the two estimators.
    differences = frame.assign(delta=frame.CellSR-frame.soft).groupby(["seed","theta"]).delta.mean().unstack()
    draws = np.random.default_rng(20261124).integers(len(differences), size=(20000,len(differences)))
    summary["delta"] = differences.mean()
    summary["lo"], summary["hi"] = np.quantile(differences.to_numpy()[draws].mean(axis=1), [.025,.975],axis=0)
    summary.to_csv(destination / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    gap = summary.index.to_numpy() - 1.2
    for method, name, marker in zip(metrics, ["Independent", "Joint", "Soft sharing", "CellSR"], ["o","s","^","D"]):
        axes[0].plot(gap, summary[method], marker=marker, label=name)
    axes[0].set_ylabel("Target coefficient MAE")
    axes[0].legend(frameon=False)
    axes[1].plot(gap, 100 * summary.accepted / 128, marker="o", color="#266478")
    axes[1].set_ylabel("Accepted equalities (%)")
    for ax in axes:
        ax.set_xlabel("Auxiliary minus target coefficient")
        ax.grid(axis="y", alpha=.25)
    fig.savefig(destination / "coefficient_gap.pdf")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
