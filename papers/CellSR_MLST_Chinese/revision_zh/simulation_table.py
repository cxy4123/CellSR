"""Refit four coefficient-sharing rules on identical archived heterogeneous structures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/cellsr1_revision"))
from search_path_ablation import ALL, OUT, dataset, label, law
from cellsr.joint_information import declared_star_links, fit_linear_links, select_declared_links, validation_losses


def main():
    archives = pd.read_csv(OUT / "ablation.csv").query("environment=='heterogeneous'")
    keys = ["seed", "family", "link"]
    reference = archives.query("method=='B'").set_index(keys)
    paths = {label(path): path for path in ALL}
    rows = []
    for record in archives.query("method=='A'").itertuples():
        key = (record.seed, record.family, record.link)
        target, donor = dataset(record.seed, record.family, record.environment, record.link)
        laws = [law("target", target, paths[record.formula], 2),
                law("donor", donor, paths[record.donor_formula], 2)]
        links = declared_star_links(laws)
        independent = fit_linear_links(laws)
        joint = fit_linear_links(laws, links)
        options = [independent] + [fit_linear_links(laws, links, penalty=l)
                                   for l in (.001, .01, .1, 1, 10, 100, 1000)] + [joint]
        soft = min(options, key=lambda fit: validation_losses(laws, fit).mean())
        selected = select_declared_links(laws)
        estimates = dict(independent=independent, joint=joint, soft=soft, CellSR=selected.estimate)
        errors = {name: abs(fit.coefficients["target"][1]-1.2) for name, fit in estimates.items()}
        np.testing.assert_allclose([errors["independent"], errors["CellSR"]],
                                   [record.theta_error, reference.loc[key,"theta_error"]], atol=1e-10)
        rows.append(dict(zip(keys,key), **errors, accepted=bool(selected.links)))
    frame = pd.DataFrame(rows)
    assert len(frame)==256 and not frame.duplicated(keys).any()
    frame.to_csv(HERE / "simulation_table.csv",index=False)
    print(frame.groupby("link").mean(numeric_only=True).to_string())
    valid = frame.query("link=='valid'").groupby("seed")[["independent","CellSR"]].mean()
    delta = (valid.independent-valid.CellSR).to_numpy()
    draws = np.random.default_rng(8846).choice(delta, size=(20000,len(delta)),replace=True)
    print("paired difference (95% CI)",delta.mean(),np.quantile(draws.mean(axis=1),[.025,.975]))


if __name__=="__main__":
    main()
