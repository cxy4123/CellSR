"""Real case screen: horizontal and vertical whole-body impulse laws on normal walking."""
from pathlib import Path
from itertools import permutations
import hashlib
import re
import zipfile
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from paderborn_blind_fit import lawset,choose,edgewise

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.cellsr1_revision.weighted_gait_signals import DATA,trial_windows
OUT=ROOT/"benchmark_outputs/cellsr1_revision/real_case_screen_mass"

def extract():
    expression=re.compile(r"OpenSim/(SUBJ\d+)/trcFiles/((?:SUBJ\d+)_NormalWalking_(\d+))_markers\.trc")
    rows=[];log=[]
    with zipfile.ZipFile(DATA/"OpenSim.zip") as archive:
        for member in archive.namelist():
            match=expression.fullmatch(member)
            if not match:continue
            subject,stub,n=match.groups()
            try:
                win=trial_windows(archive.read(member),archive.read(f"OpenSim/{subject}/forceFilesOriginal/{stub}_forces_COP_original.mot"))
            except (ValueError,KeyError) as e:
                log.append((subject,int(n),str(e)));continue
            rows.extend(dict(subject=subject,trial=int(n),**r) for r in win)
            log.append((subject,int(n),"included"))
    pd.DataFrame(log,columns=["subject","trial","qc"]).to_csv(OUT/"trial_qc.csv",index=False)
    d=pd.DataFrame(rows);d.to_csv(OUT/"normal_windows.csv",index=False)
    return d
def bank(d,subject,donor,fold):
    g=d[d.subject==subject]
    trials=sorted(g.trial.unique())[int(donor)::2]
    g=g[g.axis==("up" if donor else "ap")]
    pack={}
    for role,offset in [("fit",2),("val",1),("test",0)]:
        ids=trials[(fold+offset)%3::3]
        q=g[g.trial.isin(ids)];assert len(q)>1
        pack[role]=np.column_stack([70*q.xM,q.J])
    return pack
def compare(d,out):
    eligible=sorted(s for s,g in d.groupby("subject") if g.trial.nunique()>=6)
    print("eligible",eligible,flush=True)
    assert len(eligible)>=2
    rows=[];keys=[];rng=np.random.default_rng(9468)
    banks={(s,b,f):bank(d,s,b,f) for s in eligible for b in (False,True) for f in range(3)}
    for s,alt in permutations(eligible,2):
        for fold in range(3):
            donors=[s,alt] if rng.integers(2) else [alt,s];case=len(keys)
            parts=[banks[s,False,fold],*[banks[k,True,fold] for k in donors]]
            data={f"{a}_{r}":v for a,p in zip("ABC",parts) for r,v in p.items()}
            laws=lawset(data); full=lawset(data,True)
            options=choose(laws);beta=edgewise(laws,(0,0)).coefficients
            nearest=min("BC",key=lambda c:abs(beta[c].item()-beta["A"].item()))
            options["nearest"]=(np.inf,0) if nearest=="B" else (0,np.inf)
            for m,p in options.items():
                est=edgewise(full,p)
                rows.append(dict(case=case,method=m,pred=70*est.coefficients["A"].item(),B=p[0],C=p[1]))
            keys.append(dict(case=case,subject=s,alt=alt,fold=fold,same="BC"[donors.index(s)]))
    r=pd.DataFrame(rows);file=out/"predictions_before_scoring.csv";r.to_csv(file,index=False)
    (out/"prediction_hash.txt").write_text(hashlib.sha256(file.read_bytes()).hexdigest())
    meta=loadmat(DATA/"MAT_normalizedData-v21Aug2026.mat",simplify_cells=True)["data"]
    ref={s:meta[f"SUBJ{int(s[4:])}"]["char"]["mass"] for s in eligible}
    r=r.merge(pd.DataFrame(keys),on="case",validate="many_to_one")
    r["mae"]=(r.pred-r.subject.map(ref)).abs();r["difference"]=(r.subject.map(ref)-r.alt.map(ref)).abs()
    r["exact"]=np.where(r.same=="B",(r.B==np.inf)&(r.C==0),(r.C==np.inf)&(r.B==0))
    r["both"]=r.exact&(r.mae<r.case.map(r[r.method=="independent"].set_index("case").mae)-1e-8)
    o=r[r.method=="fixed_"+r.same].copy();o["method"]="known_single";r=pd.concat([r,o])
    r.to_csv(out/"scored.csv",index=False)
    summary=[]
    for kind,g in [("all",r),("within_5kg",r[r.difference<=5]),("over_5kg",r[r.difference>5])]:
        by=g.groupby(["subject","method"])[["mae","exact","both"]].mean()
        score=by.groupby("method").mean();score["cases"]=g.case.nunique();score["subjects"]=g.subject.nunique();score["stratum"]=kind
        summary.append(score)
    pd.concat(summary).to_csv(out/"summary.csv")
    print(summary[0].round(3).to_string())
if __name__=="__main__":compare(extract(),OUT)

