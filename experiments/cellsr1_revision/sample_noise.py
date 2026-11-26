"""One-factor scans of sample size and Gaussian noise with fresh local search.

Target/auxiliary slopes 1.2/1.2 or 1.2/1.3. Two families x 64 seeds.
At sigma=.28 vary target n=(8,12,24,32) at auxiliary n=96, and auxiliary
n=(24,48,96,192) at target n=12. At 12/96 vary sigma=(.14,.28,.56).
Smaller training sets are prefixes and larger ones append new independent
draws. Validation inputs and standard-normal noise realizations are fixed.
All pairwise MAE contrasts are CellSR minus comparator; bootstrap seeds,
keeping both families together. Intervals are pointwise percentile intervals.
"""
from functools import lru_cache
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from search_path_ablation import (ALL, PARENTS, OPS, OUT, FAMILIES, SEEDS,
                                  dataset, feature, label, law)
from cellsr.joint_information import declared_star_links, fit_linear_links, select_declared_links, validation_losses


def observations(seed, family, side, nfit, theta, sigma, original):
    path = FAMILIES[family] if side==0 else (("cos",) if family=="even" else ("square",))
    def response(x,z):
        return 1.2*x + (.85 if side==0 else -1.1)*feature(path,z) + (.3 if side==0 else -.4)
    out = []
    for i,(x,z,y) in enumerate(original[:3]):
        if sigma != .28:
            mean = response(x,z)
            y = mean + (sigma/.28)*(y-mean)
        if i==0 and nfit>len(x):
            rng = np.random.default_rng([seed,side,20261124])
            # Draw the maximum extension first, so 24 and 32 share a prefix.
            nmax = (32 if side==0 else 192)-len(x)
            ze = rng.uniform(-1.4,1.4,nmax)
            u = rng.normal(size=nmax)
            xe = .82*ze+.32*u if side==0 else .2*ze+u
            ye = response(xe,ze)+sigma*rng.normal(size=nmax)
            x,z,y = np.r_[x,xe],np.r_[z,ze],np.r_[y,ye]
        if i==0:
            x,z,y = x[:nfit],z[:nfit],y[:nfit]
        out.append((x,z,y+(theta-1.2)*x))
    return out


def evaluate(target, donor):
    def donor_score(path):
        laws = [law("donor",donor,path,1)]
        return validation_losses(laws,fit_linear_links(laws))[0]
    dp = min(ALL,key=donor_score)
    dl = law("donor",donor,dp,1)
    @lru_cache(None)
    def target_score(path):
        laws = [law("target",target,path,1),dl]
        return validation_losses(laws,fit_linear_links(laws)).mean()
    parents = sorted(PARENTS,key=target_score)[:2]
    tp = min([()]+PARENTS+[p+(op,) for p in parents for op in OPS],key=target_score)
    laws = [law("target",target,tp,2),law("donor",donor,dp,2)]
    links = declared_star_links(laws)
    ind = fit_linear_links(laws)
    joint = fit_linear_links(laws,links)
    candidates = [ind]+[fit_linear_links(laws,links,penalty=p) for p in (.001,.01,.1,1,10,100,1000)]+[joint]
    soft = min(candidates,key=lambda fit: validation_losses(laws,fit).mean())
    selected = select_declared_links(laws)
    errors = {k:abs(v.coefficients["target"][1]-1.2) for k,v in
              dict(independent=ind,joint=joint,soft=soft,CellSR=selected.estimate).items()}
    return dict(**errors,accepted=bool(selected.links),formula=label(tp),donor_formula=label(dp))


def main():
    dest = OUT.parent/"sample_noise"
    dest.mkdir(exist_ok=True)
    configs = sorted({(n,96,.28) for n in (8,12,24,32)} |
                     {(12,n,.28) for n in (24,48,96,192)} | {(12,96,s) for s in (.14,.28,.56)})
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            base = dataset(seed,family,"heterogeneous","valid")
            for nt,na,sigma in configs:
                target = observations(seed,family,0,nt,1.2,sigma,base[0])
                for theta in (1.2,1.3):
                    donor = observations(seed,family,1,na,theta,sigma,base[1])
                    rows.append(dict(nt=nt,na=na,sigma=sigma,theta=theta,family=family,seed=seed,
                                     **evaluate(target,donor)))
        print("Finished",family,flush=True)
    df = pd.DataFrame(rows)
    keys = ["seed","family","theta"]
    ref = pd.read_csv(OUT.parent/"coefficient_gap/cases.csv").query("theta<=1.3").set_index(keys).sort_index()
    replay = df.query("nt==12 and na==96 and sigma==.28").set_index(keys).sort_index()
    assert ref.index.equals(replay.index) and replay.accepted.equals(ref.accepted)
    metrics = ["independent","joint","soft","CellSR"]
    np.testing.assert_allclose(replay[metrics],ref[metrics],rtol=0,atol=1e-10)
    groups = ["nt","na","sigma","theta"]
    assert len(df)==2304 and not df.duplicated(groups+["family","seed"]).any()
    df.to_csv(dest/"cases.csv",index=False)
    summary = df.groupby(groups)[metrics+["accepted"]].mean()
    draws = np.random.default_rng(20261124).integers(64,size=(20000,64))
    for key,g in df.groupby(groups):
        v = g.groupby("seed")[metrics].mean()
        for name in ("independent","soft"):
            diff = (v.CellSR-v[name]).to_numpy()
            lo,hi = np.quantile(diff[draws].mean(axis=1),[.025,.975])
            summary.loc[key,[name+"_delta",name+"_lo",name+"_hi"]] = [diff.mean(),lo,hi]
    summary.to_csv(dest/"summary.csv")
    table = summary.reset_index()
    fig,axes = plt.subplots(2,3,figsize=(11,6),layout="constrained",sharey=True)
    for row,theta in enumerate((1.2,1.3)):
        subset = table[table.theta==theta]
        for ax,(x,q,title) in zip(axes[row],[
            ("nt","na==96 and sigma==.28","Target training points"),
            ("na","nt==12 and sigma==.28","Auxiliary training points"),
            ("sigma","na==96 and nt==12","Noise standard deviation")]):
            g = subset.query(q).sort_values(x)
            for method in ("independent","soft","CellSR"):
                ax.plot(g[x],g[method],marker="o",label=method)
            ax.set_xticks(g[x]); ax.set_xlabel(title); ax.grid(axis="y",alpha=.25)
        axes[row,0].set_ylabel(f"Auxiliary slope {theta:.1f}\nTarget coefficient MAE")
    axes[0,0].legend()
    fig.savefig(dest/"sample_noise.pdf")
    print(summary.round(4).to_string())


if __name__=="__main__":
    main()
