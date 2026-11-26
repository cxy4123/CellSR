"""KAt blind experiment shared paths and integrity helpers; no raw-data dependencies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"benchmark_outputs/cellsr1_revision/paderborn_blind_mixed_v1"
DATA=ROOT/"data/cellsr_paderborn_blind"
BLIND=OUT/"blind_features"
KEY=DATA/"custody.csv"
def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()
def stamp(): return datetime.now(timezone.utc).isoformat()
def write_new(path, value):
    with Path(path).open("x",encoding="utf8") as stream: json.dump(value,stream,indent=2)
def protocol_check():
    lock=json.loads((OUT/"protocol_lock.json").read_text(encoding="utf-8-sig"))
    assert sha(OUT/"protocol.json").upper()==lock["protocol_sha256"].upper()
