"""Calibrate nearest-neighbour rejection on Horst; evaluate fixed CellSR B on Antwerp."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from external_pair_compare import OUT as HORST_OUT, ROOT, bank, lawset, edgewise
from cellsr.joint_information import select_declared_links

OUT = ROOT / "benchmark_outputs/cellsr1_revision/matched_false_share"


def predictor(windows, folds=3):
    @lru_cache(None)
    def arm(s, donor, fold):
        if folds == 3:
            data = bank(windows, s, donor, fold)
        else:
            g = windows[windows.subject == s]
            trials = sorted(g.trial.unique())[int(donor)::2]
            g = g[g.axis == ("up" if donor else "ap")]
            data = {}
            for role, offset in [("fit",fold),("val",1-fold)]:
                part = g[g.trial.isin(trials[offset::2])]
                assert len(part) >= 2
                data[role] = np.column_stack([70*part.xM,part.J])
        def slope(role):
            x, y = data[role].T
            return x @ y / (x @ x)
        return data, slope("fit"), abs(slope("fit") - slope("val"))
    return arm


def nearest(inputs):
    gap = np.abs(np.array([item[1] for item in inputs[1:]]) - inputs[0][1])
    j = int(np.argmin(gap))
    denominator = inputs[0][2] + inputs[j+1][2]
    return "BC"[j], gap[j] / denominator if denominator else (0.0 if gap[j] == 0 else np.inf)


def boundary(frame):
    zero = frame[frame.kind == "zero"].sort_values(["score", "tie"])
    n = int((zero.B != "-").sum())
    if n == 0:
        return [-1., ""]
    weights = 1 / zero.groupby("target").target.transform("size")
    target = weights[zero.B != "-"].sum()
    cumulative = weights.cumsum().to_numpy()
    count = int(np.argmin(abs(np.r_[0,cumulative] - target)))
    if not count:
        return [-1., ""]
    last = zero.iloc[count-1]
    return [float(last.score), last.tie]


def tie(case):
    return hashlib.sha256(str(case).encode()).hexdigest()


def calibrate():
    inputs = predictor(pd.read_csv(HORST_OUT / "gait_windows.csv"))
    keys = pd.read_csv(HORST_OUT / "gait/key_scored.csv")
    preds = pd.read_csv(HORST_OUT / "gait/predictions.csv")
    keys["B"] = keys.case.map(preds[preds.method == "CellSR_B"].set_index("case").graph)
    zero = keys[keys.kind == "zero"].copy()
    zero["score"] = [nearest([inputs(r.target, False, r.fold), inputs(r.donor_b, True, r.fold),
                              inputs(r.donor_c, True, r.fold)])[1] for r in zero.itertuples()]
    zero["tie"] = zero.case.map(tie)
    threshold = boundary(zero)
    result = dict(threshold=threshold, n=len(zero), false_B=int((zero.B != "-").sum()),
                  false_nn=int(accept(zero, threshold).sum()))
    assert result["false_B"] == result["false_nn"]
    (OUT / "calibration.json").write_text(json.dumps(result, indent=2))
    print(result)


def accept(frame, threshold):
    return (frame.score < threshold[0]) | ((frame.score == threshold[0]) & (frame.tie <= threshold[1]))


def external():
    windows = pd.read_csv(OUT / "antwerp_windows.csv")
    inputs = predictor(windows, folds=2)
    valid = []
    for subject, group in windows.groupby("subject"):
        if group.trial.nunique() < 4:
            continue
        try:
            for donor in (False, True):
                for fold in range(2):
                    inputs(subject, donor, fold)
        except AssertionError:
            continue
        valid.append(subject)
    assert len(valid) >= 3, valid
    rng = np.random.default_rng(91229)
    rows = []
    for target in valid:
        others = rng.permutation([s for s in valid if s != target]).tolist()
        for i, other in enumerate(others):
            for f in range(2):
                for kind, donors in [("one", [target, other]), ("zero", [other, others[(i+1)%len(others)]])]:
                    b,c = rng.permutation(donors)
                    arms = [inputs(target,False,f),inputs(b,True,f),inputs(c,True,f)]
                    data = {k+"_"+role: arr for k, arm in zip("ABC",arms) for role,arr in arm[0].items()}
                    laws = lawset(data)
                    B = "".join(e.right for e in select_declared_links(laws).links) or "-"
                    nn, score = nearest(arms)
                    full = lawset(data,True)
                    predictions = {s: 70*edgewise(full,[np.inf if k in s else 0 for k in "BC"]).coefficients["A"].item()
                                   for s in ("-","B","C","BC")}
                    rows.append(dict(target=target,b=b,c=c,fold=f,kind=kind,truth="B" if target==b else "C" if target==c else "-",
                                     B=B,nn=nn,score=score,tie=tie(len(rows)),**{"m"+k:v for k,v in predictions.items()}))
    path = OUT / "sealed_predictions.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
    (OUT/"sealed_predictions.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def score():
    path = OUT / "sealed_predictions.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (OUT/"sealed_predictions.sha256").read_text()
    frame = pd.read_csv(path)
    ref = pd.read_csv(OUT/"publisher_mass.csv").set_index("subject").mass
    frame["ref"] = frame.target.map(ref)
    assert np.isfinite(frame.ref).all()
    deployed = json.loads((OUT/"calibration.json").read_text())["threshold"]
    def evaluate(df):
        matched = boundary(df)
        result = []
        for method,graph in [("B",df.B),("NN_frozen",df.nn.where(accept(df,deployed),"-")),
                             ("NN_equal",df.nn.where(accept(df,matched),"-")),("independent",pd.Series("-",index=df.index))]:
            value = np.select([graph==k for k in ("B","C","BC")],[df.mB,df.mC,df.mBC],default=df["m-"])
            error = abs(value-df.ref)
            exact = df.truth==graph
            one,zero=df.kind=="one",df.kind=="zero"
            mean = lambda x,mask: x[mask].groupby(df.target[mask]).mean().mean()
            gain=exact & (error<abs(df["m-"]-df.ref)-1e-8)
            result.append(dict(method=method,wrong_rate=mean(graph!="-",zero),one_exact=mean(exact,one),
                               both=mean(gain,one),mae=(mean(error,one)+mean(error,zero))/2,
                               n_zero=zero.sum(),false_n=((graph!="-")&zero).sum(),both_n=(gain&one).sum(),n_one=one.sum()))
        return result
    main = pd.DataFrame(evaluate(frame))
    print(main.to_string(index=False))
    main.to_csv(OUT/"comparison.csv",index=False)
    pd.DataFrame([dict(removed=s,**r) for s in frame.target.unique() for r in evaluate(frame[(frame.target!=s)&(frame.b!=s)&(frame.c!=s)])]).to_csv(OUT/"delete_person.csv",index=False)


if __name__ == "__main__":
    import sys
    {"calibrate":calibrate,"external":external,"score":score}[sys.argv[1]]()
