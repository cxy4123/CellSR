"""After scoring, cross-check publisher mass versions against quiet-standing forces."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import numpy as np
import pandas as pd
from external_pair_compare import HORST, OUT, ROOT


def static_forces(archive):
    result = {}
    for name in archive.namelist():
        if "_static_" not in name.lower() or not name.lower().endswith(".tsv"):
            continue
        subject, trial = Path(name).name.upper().split("_")[:2]
        body = archive.read(name)
        table = BytesIO(body[body.index(b"Sample\tTime"):])
        result[(subject, trial)] = np.loadtxt(table, skiprows=1, usecols=(0, 4))
    return result


def main():
    with ZipFile(HORST / "Gait_rawdata_f_1_tsv.zip") as left, ZipFile(HORST / "Gait_rawdata_f_2_tsv.zip") as right:
        first, second = static_forces(left), static_forces(right)
    rows = []
    for subject, trial in sorted(first.keys() & second.keys()):
        a, b = first[subject, trial], second[subject, trial]
        if len(a) != len(b) or not np.array_equal(a[:, 0], b[:, 0]):
            continue
        mass = (a[:, 1] + b[:, 1]) / 9.80665
        rows.append((subject, trial, np.median(mass)))
    sums = pd.DataFrame(rows, columns=["subject", "trial", "static_mass"])
    sums.to_csv(OUT / "static_force_mass_trials.csv", index=False)
    key = pd.read_csv(OUT / "gait/key_scored.csv")
    by_person = pd.DataFrame(index=sorted(key.target.unique()))
    by_person.index.name = "subject"
    by_person["static_mass"] = sums.groupby("subject").static_mass.median()
    for version, filename in [(2, "Gait_subject_info.csv"), (3, "Gait_subject_info_v3.csv")]:
        ref = pd.read_csv(HORST / filename, sep=";", index_col=0)
        by_person[f"ref_v{version}"] = ref["body mass [kg]"].astype(str).str.replace(",", ".", regex=False).astype(float)
    windows = pd.read_csv(OUT / "gait_windows.csv")
    for axis in ("ap", "up"):
        selected = windows[windows.axis == axis]
        numerator = (selected.xM * selected.J).groupby(selected.subject).sum()
        denominator = (selected.xM ** 2).groupby(selected.subject).sum()
        by_person[axis] = numerator / denominator
    by_person.to_csv(OUT / "mass_reference_audit.csv")
    for col in ("static_mass", "ap", "up"):
        for ver in ("ref_v2", "ref_v3"):
            pair = by_person[[ver, col]].dropna()
            print(col, ver, len(pair), "r", pair[ver].corr(pair[col]), "MAE", abs(pair[ver] - pair[col]).mean())
    source = HORST / "Gait_subject_info_v3.csv"
    manifest = OUT / "input_hashes.json"
    files = json.loads(manifest.read_text())
    files[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
