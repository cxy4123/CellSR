"""Paired gate ablation. See gate_protocol.md for estimands and BIC assumptions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from search_path_ablation import ALL, OUT, dataset, label, law
from cellsr.joint_information import (declared_star_links, fit_linear_links,
                                      validation_eligible, validation_losses)


def bic_choice(laws):
    a, b = laws
    x = block_diag(a.fit_x, b.fit_x)
    y = np.r_[a.fit_y, b.fit_y]
    shared = x[:, [0, 1, 2, 3, 5]].copy()
    shared[:, 1] += x[:, 4]
    models = []
    for design in (x, shared):
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        assert rank == design.shape[1]
        rss = np.sum((y - design @ beta)**2)
        models.append((len(y)*np.log(rss/len(y)) + (rank+1)*np.log(len(y)), beta[1]))
    chosen = models[1][0] < models[0][0]
    return chosen, abs(models[int(chosen)][1] - 1.2)


def main():
    dest = OUT.parent / "gate_sensitivity"
    dest.mkdir(exist_ok=True)
    old = pd.read_csv(OUT.parent / "coefficient_gap/cases.csv").set_index(["seed","family","theta"])
    cases = pd.read_csv(OUT / "ablation.csv").query("environment=='heterogeneous' and method=='A' and link=='valid'")
    paths = {label(p): p for p in ALL}
    rows = []
    factors = [1., 1.02, 1.05, 1.10, 1.20]
    for case in cases.itertuples():
        target, donor = dataset(case.seed, case.family, "heterogeneous", "valid")
        for theta in (1.2, 1.3, 1.4, 1.6, 1.8, 2.1):
            shifted = [(x,z,y+(theta-1.2)*x) for x,z,y in donor]
            laws = [law("target",target,paths[case.formula],2), law("donor",shifted,paths[case.donor_formula],2)]
            ind = fit_linear_links(laws)
            joint = fit_linear_links(laws,declared_star_links(laws))
            assert joint.diagnostics["identifiable"]
            base, candidate = [validation_losses(laws, est) for est in (ind,joint)]
            errors = [abs(est.coefficients["target"][1]-1.2) for est in (ind,joint)]
            ref = old.loc[case.seed,case.family,theta]
            decisions = {f"{factor:.2f}": validation_eligible(base,candidate,factor-1) for factor in factors}
            assert decisions["1.05"] == ref.accepted
            np.testing.assert_allclose([*errors,errors[int(ref.accepted)]],[ref.independent,ref.joint,ref.CellSR],atol=1e-10,rtol=0)
            decisions["mean_only"] = candidate.mean() < base.mean()-1e-12
            decisions["BIC"], mle = bic_choice(laws)
            for method, accepted in decisions.items():
                rows.append(dict(seed=case.seed,family=case.family,theta=theta,method=method,accepted=accepted,
                                 error=errors[int(accepted)],bic_mle=mle if method=="BIC" else np.nan,
                                 default=ref.CellSR,soft=ref.soft,independent=ref.independent))
    frame = pd.DataFrame(rows)
    assert len(frame)==768*7 and not frame.duplicated(["seed","family","theta","method"]).any()
    frame.to_csv(dest/"cases.csv",index=False)
    frame["delta"] = frame.error-frame.default
    summary = frame.groupby(["theta","method"]).agg(accepted=("accepted","sum"),rate=("accepted","mean"),
            mae=("error","mean"),delta=("delta","mean"),soft=("soft","mean"),independent=("independent","mean"),
            bic_mle=("bic_mle","mean"))
    paired = frame.groupby(["seed","theta","method"]).delta.mean().unstack(["theta","method"])
    draws = np.random.default_rng(20261124).integers(64,size=(20000,64))
    for col in paired:
        lo,hi = np.quantile(paired[col].to_numpy()[draws].mean(axis=1),[.025,.975])
        summary.loc[col,["lo","hi"]] = [lo,hi]
    summary.to_csv(dest/"summary.csv")
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for theta, group in summary.groupby(level="theta"):
        sel = group.droplevel("theta").loc[[f"{x:.2f}" for x in factors]]
        for ax,metric in zip(axes,["rate","mae"]):
            ax.plot(factors,sel[metric],marker="o",ms=3,label=f"{theta:g}")
    for ax in axes:
        ax.set_xlabel("Per-equation validation loss multiplier")
        ax.axvline(1.05,ls=":",color="grey")
        ax.set_xticks(factors)
    axes[0].set_ylabel("Fraction accepting equality")
    axes[1].set_ylabel("Target coefficient MAE")
    axes[0].legend(title="Auxiliary coefficient",fontsize=8,ncol=2)
    fig.savefig(dest/"thresholds.pdf")
    print(summary.loc[(slice(None),["1.05","mean_only","BIC"]),:].round(4).to_string())
    print(summary.accepted.unstack().to_string())


if __name__=="__main__":
    main()
