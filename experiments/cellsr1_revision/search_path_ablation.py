"""Matched search-path ablation for a declared scalar link in two symbolic laws."""
from pathlib import Path
from itertools import product
import argparse
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cellsr.expression import ExpressionNode
from cellsr.joint_information import (LinearLaw, declared_star_links, fit_linear_links,
                                      validation_losses, validation_eligible)

OUT = ROOT / "benchmark_outputs/cellsr1_revision/search_path_classsr_v1"
OPS = ("sin", "cos", "square", "cube")
PARENTS = [(op,) for op in OPS]
ALL = [()] + PARENTS + list(product(OPS, repeat=2))
FAMILIES = {"even": ("square", "sin"), "odd": ("cube", "sin")}
SEEDS = list(range(39101, 39165))
PROTOCOL = dict(seeds=SEEDS, neighbor_seeds=SEEDS[:12], families=FAMILIES,
                environments=["heterogeneous", "common"], links=["valid", "conflict"],
                target_counts=[12, 16, 16, 512], donor_counts=[96, 48, 48, 512],
                search="4 unary parents; expand best 2 with each of 4 unary operators; include z",
                donor="independent validation choice among 21 trees",
                primary="paired target structure recovery C minus B; average conditions within seed",
                secondary="C minus terminal rerank on B paths; theta absolute error; target clean OOD RMSE",
                neighbor="official PhySO ClassSR, common only, 4 epochs x 16 programs; training only optimizer; search validation selection",
                final_gate="fresh validation, train-only coefficients retained")

def node(path):
    tree = ExpressionNode.variable("z")
    for op in path:
        tree = ExpressionNode.op(op, [tree])
    return tree

def feature(path, z):
    return node(path).evaluate({"z":z})

def label(path):
    return str(node(path).to_sympy())

def equivalent(first, second):
    # A private slope/intercept makes affine transforms of the feature equivalent.
    grid = np.linspace(-2.71, 2.71, 257)
    reference = feature(second, grid)
    basis = np.column_stack([np.ones(len(grid)), feature(first, grid)])
    residual = reference - basis @ np.linalg.lstsq(basis, reference, rcond=None)[0]
    return bool(np.max(np.abs(residual)) < 1e-8)

def dataset(seed, family, environment, link):
    rng = np.random.default_rng(seed)
    truth = FAMILIES[family]
    laws = []
    for i, counts in enumerate([PROTOCOL["target_counts"], PROTOCOL["donor_counts"]]):
        nuisance = truth if i == 0 or environment == "common" else (("cos",) if family == "even" else ("square",))
        theta = 1.2 if i == 0 or link == "valid" else 2.1
        parts = []
        for k, n in enumerate(counts):
            z = rng.uniform(-1.4 if k < 3 else -2.4, 1.4 if k < 3 else 2.4, n)
            u = rng.normal(size=n)
            x = (.82*z + .32*u if i == 0 else .2*z + u) if k < 3 else u
            mean = theta*x + (0.85 if i == 0 else -1.1)*feature(nuisance,z) + (.3 if i == 0 else -.4)
            eps = rng.normal(0,.28,n)
            parts.append((x,z,mean+eps if k<3 else mean))
        laws.append(parts)
    return laws

def design(part, path):
    x,z,y = part
    return np.column_stack([np.ones(len(x)), x, feature(path,z)])

def law(name, parts, path, val):
    return LinearLaw(name,design(parts[0],path),parts[0][2],
                     design(parts[val],path),parts[val][2],{"theta":1})

def run_case(seed, family, environment, link):
    data = dataset(seed, family, environment, link)
    target, donor = data
    def donor_score(path):
        model = law("donor", donor, path, 1)
        return validation_losses([model], fit_linear_links([model]))[0]
    donor_path = min(ALL, key=donor_score)
    cache = {}
    def evaluate(path, val=1):
        key = path,val
        if key not in cache:
            laws = [law("target",target,path,val), law("donor",donor,donor_path,val)]
            ind = fit_linear_links(laws)
            joint = fit_linear_links(laws,declared_star_links(laws))
            base, shared = (validation_losses(laws, e) for e in (ind,joint))
            use = joint.diagnostics["identifiable"] and validation_eligible(base,shared)
            cache[key] = (base.mean(), (shared if use else base).mean(), ind,
                          joint if use else ind, bool(use))
        return cache[key]
    def search(coupled):
        idx = int(coupled)
        parents = sorted(PARENTS,key=lambda p:evaluate(p)[idx])[:2]
        pool = [()] + PARENTS + [parent+(op,) for parent in parents for op in OPS]
        return pool,parents
    blind,pb = search(False)
    aware,pc = search(True)
    rows = []
    for method,pool,parents,rank,gate in [
        ("A",blind,pb,0,False), ("B",blind,pb,0,True),
        ("B_rerank",blind,pb,1,True), ("C",aware,pc,1,True)]:
        path = min(pool,key=lambda p:evaluate(p)[rank])
        result = evaluate(path,2)
        coeff = result[3 if gate else 2].coefficients["target"]
        pred = design(target[3],path) @ coeff
        rows.append(dict(seed=seed,family=family,environment=environment,link=link,
                    method=method,formula=label(path),donor_formula=label(donor_path),
                    parents="|".join(p[0] for p in parents),pool_changed=set(pc)!=set(pb),
                    reached=any(equivalent(p,FAMILIES[family]) for p in pool),
                    exact=equivalent(path,FAMILIES[family]),theta=float(coeff[1]),
                    theta_error=abs(coeff[1]-1.2),rmse=np.mean((pred-target[3][2])**2)**.5,
                    shared=bool(gate and result[4]),search_candidates=len(pool),
                    donor_candidates=len(ALL),matched_solve_budget=len(ALL)+2*(len(pool)+1)))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"protocol.json").write_text(json.dumps(PROTOCOL,indent=2))
    if args.protocol_only: return
    rows=[]
    for family,env,link in product(FAMILIES,PROTOCOL["environments"],PROTOCOL["links"]):
        for seed in SEEDS:
            rows.extend(run_case(seed,family,env,link))
        print(family,env,link,flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==2048 and not df.isna().any().any()
    assert (df.search_candidates==13).all() and (df.matched_solve_budget==49).all()
    df.to_csv(OUT/"ablation.csv",index=False)

if __name__=="__main__":
    main()

