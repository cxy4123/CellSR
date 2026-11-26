"""Fit on anonymous arrays. This process receives neither the speed channel nor a recording key."""
from itertools import product
import json
import sys
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from paderborn_blind_paths import ROOT, OUT, BLIND, sha, stamp, protocol_check, write_new
sys.path.insert(0,str(ROOT))
from cellsr.joint_information import (LinearLaw, JointEstimate, declared_star_links, fit_linear_links,
                                      select_declared_links, validation_losses, validation_eligible)

GRID=[0,.001,.01,.1,1,10,100,1000,np.inf]
EDGE_GRID=[0,.01,1,100,np.inf]

def edgewise(laws,strengths):
    # Three scalar sufficient statistics give the same weighted least-squares objective.
    gram=[]; rhs=[]
    for law in laws:
        x=law.fit_x[:,0];y=law.fit_y
        w=1/(len(y)*max(y.var(),1e-24))
        gram.append((x@x)*w);rhs.append((x@y)*w)
    matrix=np.diag(gram);hard=[]
    for j,value in enumerate(strengths):
        contrast=np.zeros(3);contrast[0]=1;contrast[j+1]=-1
        if np.isinf(value): hard.append(contrast)
        else: matrix+=value*np.outer(contrast,contrast)
    z=null_space(np.array(hard)) if hard else np.eye(3)
    coeff=z@np.linalg.solve(z.T@matrix@z,z.T@rhs)
    return JointEstimate({arm:np.array([c]) for arm,c in zip("ABC",coeff)})

def lawset(data,refit=False):
    out=[]
    for arm in "ABC":
        train=data[f"{arm}_fit"];val=data[f"{arm}_val"]
        if refit: train=np.vstack([train,val])
        out.append(LinearLaw(arm,train[:,:1],train[:,1],val[:,:1],val[:,1],{"f":0}))
    return out

def choose(laws):
    links=declared_star_links(laws)
    hard={};soft={}
    for pair in product([0,np.inf],repeat=2):
        selected=tuple(e for e,s in zip(links,pair) if s)
        est=fit_linear_links(laws,selected)
        hard[pair]=validation_losses(laws,est)
    independent=(0,0);joint=(np.inf,np.inf)
    base=hard[independent]
    for strength in GRID:
        est=fit_linear_links(laws,links if strength else (),None if np.isinf(strength) or strength==0 else strength)
        soft[strength]=validation_losses(laws,est)
        np.testing.assert_allclose([v.item() for v in edgewise(laws,(strength,strength)).coefficients.values()],
                                   [v.item() for v in est.coefficients.values()],rtol=1e-8,atol=1e-8)
    selected=select_declared_links(laws,links)
    selected_pair=tuple(np.inf if l in selected.links else 0 for l in links)
    def prefer(pool):return min(pool,key=lambda p:pool[p].mean())
    soft_sel=prefer(soft)
    guarded=prefer({k:v for k,v in soft.items() if k==0 or validation_eligible(base,v)})
    dense={pair:validation_losses(laws,edgewise(laws,pair)) for pair in product(EDGE_GRID,repeat=2)}
    return {"independent":independent,"joint":joint,"global_soft":(soft_sel,soft_sel),
            "guarded_soft":(guarded,guarded),"validation_only":prefer(hard),"CellSR_B":selected_pair,
            "edgewise_soft":prefer(dense),"fixed_B":(np.inf,0),"fixed_C":(0,np.inf)}

def main():
    protocol_check()
    prep=json.loads((OUT/"preparation.json").read_text())
    specs=json.loads((OUT/"implementation_lock.json").read_text())
    for n,h in specs["sources"].items(): assert sha(ROOT/n)==h
    preds=OUT/"predictions";preds.mkdir(exist_ok=True)
    rows=[];hashes={}
    for filename,digest in sorted(prep["features"].items()):
        path=BLIND/filename
        assert sha(path)==digest
        with np.load(path,allow_pickle=False) as data:
            laws=lawset(data); combined=lawset(data,True)
            chosen=choose(laws)
            saved={}
            for method,pair in chosen.items():
                # Native exact fitting for binary configurations, numerical equivalent for mixed strengths.
                if all(v in (0,np.inf) for v in pair):
                    links=tuple(l for l,v in zip(declared_star_links(combined),pair) if v)
                    est=fit_linear_links(combined,links)
                else: est=edgewise(combined,pair)
                entry=dict(case=path.stem,method=method,penalty_B=pair[0],penalty_C=pair[1])
                for arm in "ABC":
                    beta=est.coefficients[arm].item()
                    saved[f"{method}_{arm}"]=data[f"{arm}_test"][:,0]*beta
                    entry["rpm_"+arm]=60*beta
                rows.append(entry)
            target=preds/filename
            np.savez_compressed(target,**saved);hashes[filename]=sha(target)
    p=OUT/"blind_estimates.csv";pd.DataFrame(rows).to_csv(p,index=False)
    write_new(OUT/"prediction_seal.json",dict(utc=stamp(),estimates=sha(p),predictions=hashes,
              protocol=sha(OUT/"protocol.json"),clarification=sha(OUT/"metadata_clarification.txt"),
              implementation=sha(OUT/"implementation_lock.json"),prepared=sha(OUT/"preparation.json")))
    print("Sealed",len(hashes),"anonymous cases",flush=True)
if __name__=="__main__": main()
