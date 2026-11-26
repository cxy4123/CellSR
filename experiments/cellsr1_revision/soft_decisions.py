"""Eight-fold, seed-grouped comparison of hard decisions and soft coefficient gaps.

Primary: deterministic inclusive cutoff closest to CellSR's pooled false-positive
count in seven calibration folds (equal weights for the five nonzero gaps).
Include reject-all, break distance ties conservatively. No tie splitting.
Secondary: match calibration risk exactly in expectation by randomizing only
at the boundary. Report expected rates analytically. Fold assignments are
fixed before scoring; leave coefficient estimates unchanged for every rule.
"""
import numpy as np
import pandas as pd

from coefficient_gap import soft_fit
from search_path_ablation import ALL, OUT, SEEDS, dataset, label, law
from cellsr.joint_information import declared_star_links, fit_linear_links


def scores():
    archive = pd.read_csv(OUT.parent / "coefficient_gap/cases.csv")
    formulas = pd.read_csv(OUT / "ablation.csv").query("method=='A' and link=='valid' and environment=='heterogeneous'")
    paths = {label(path):path for path in ALL}
    rows = []
    for case in formulas.itertuples():
        target, donor = dataset(case.seed, case.family, "heterogeneous", "valid")
        target_law = law("target", target, paths[case.formula], 2)
        for theta in sorted(archive.theta.unique()):
            shifted = [(x,z,y+(theta-1.2)*x) for x,z,y in donor]
            laws = [target_law, law("donor", shifted, paths[case.donor_formula], 2)]
            links = declared_star_links(laws)
            independent = fit_linear_links(laws)
            joint = fit_linear_links(laws, links)
            soft = soft_fit(laws, links, independent, joint)
            a, b = [soft.coefficients[k][1] for k in ("target","donor")]
            full = soft is joint
            if full:
                assert abs(a-b)<1e-10
            rows.append(dict(seed=case.seed,family=case.family,theta=theta,a=a,b=b,
                             score=0. if full else abs(a-b),full=full))
    df = pd.DataFrame(rows).merge(archive,on=["seed","family","theta"],validate="one_to_one")
    assert len(df)==768 and df.groupby("seed").size().eq(12).all()
    np.testing.assert_allclose(abs(df.a-1.2),df.soft,atol=1e-10,rtol=0)
    return df


def calibrate(gaps, hard, weights):
    # Exclude zero-weight bootstrap observations from candidate cutoffs.
    observed = weights>0
    levels, indices = np.unique(gaps[observed],return_inverse=True)
    counts = np.bincount(indices,weights=weights[observed])
    total = np.r_[0.,counts.cumsum()]
    target = weights @ hard
    closest = np.abs(total-target).argmin()
    cutoff = np.r_[-np.inf,levels][closest]
    j = np.searchsorted(total[1:],target)
    return cutoff, levels[j], (target-total[j])/counts[j]


def main():
    dest = OUT.parent/"soft_decisions"
    dest.mkdir(exist_ok=True)
    df = scores().sort_values(["seed","family","theta"]).reset_index(drop=True)
    rng = np.random.default_rng(20261124)
    blocks = rng.permutation(SEEDS).reshape(8,8)
    df["fold"] = df.seed.map({s:f for f,block in enumerate(blocks) for s in block})
    masks = [(df.fold.ne(f).to_numpy() & df.theta.ne(1.2).to_numpy(),df.fold.eq(f).to_numpy()) for f in range(8)]
    hard = df.accepted.to_numpy(dtype=float)
    gaps = df.score.to_numpy()
    def predict(weights, log=None):
        det, prob = np.zeros(len(df)), np.zeros(len(df))
        for f,(train,test) in enumerate(masks):
            c,b,u = calibrate(gaps[train],hard[train],weights[train])
            det[test] = gaps[test]<=c
            prob[test] = (gaps[test]<b) + u*(gaps[test]==b)
            if log is not None:
                assert np.isclose(((gaps[train]<b)+u*(gaps[train]==b)).sum(),hard[train].sum())
                log.append(dict(fold=f,cutoff=c,boundary=b,tie_probability=u,
                                train_false_target=hard[train].sum(),train_false_soft=(gaps[train]<=c).sum(),
                                train_zero_scores=(gaps[train]==0).sum()))
        return det,prob
    folds = []
    df["calibrated"],df["randomized_expected"] = predict(np.ones(len(df)),folds)
    df.to_csv(dest/"cases.csv",index=False)
    pd.DataFrame(folds).to_csv(dest/"calibration.csv",index=False)
    decision_cols = ["accepted","full","calibrated","randomized_expected"]
    report = df.groupby("theta")[decision_cols+["CellSR","soft"]].mean()
    report.to_csv(dest/"by_theta.csv")
    # Stratified paired bootstrap: resample whole seeds within folds, refit each
    # cutoff on its own resampled training folds, evaluate on held-out seeds.
    seed_index = df.seed.to_numpy()-min(SEEDS)
    truth = df.theta.to_numpy()==1.2
    def contrasts(weights,det,prob):
        return [np.average((soft-hard)[mask],weights=weights[mask])
                for soft in (det,prob) for mask in (truth,~truth)]
    values = []
    for _ in range(2000):
        sampled = rng.choice(8,size=(8,8))
        seeds = np.take_along_axis(blocks,sampled,axis=1)
        w = np.bincount(seeds.ravel()-min(SEEDS),minlength=64)[seed_index]
        values.append(contrasts(w,*predict(w)))
    bounds = np.quantile(values,[.025,.975],axis=0)
    delta = pd.DataFrame(dict(rule=["calibrated"]*2+["randomized_expected"]*2,
        metric=["TPR","pooled_FPR"]*2,
        soft_minus_CellSR=contrasts(np.ones(len(df)),df.calibrated,df.randomized_expected),
        lo=bounds[0],hi=bounds[1]))
    delta.to_csv(dest/"paired.csv",index=False)
    print(report.round(4).to_string())
    print(df.loc[~truth,decision_cols].sum().to_string())
    print(delta.round(4).to_string(index=False))
    print(pd.DataFrame(folds).round(5).to_string(index=False))


if __name__=="__main__":
    main()
