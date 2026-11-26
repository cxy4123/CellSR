"""Read only anthropometric metadata from the publisher's large MATLAB 7.3 file."""
from io import RawIOBase
from pathlib import Path
import h5py
import pandas as pd
import requests
from matched_false_share import ROOT, OUT


class FigshareReader(RawIOBase):
    size = 6244593847
    block = 256*1024
    def __init__(self):
        self.pos = 0
        self.cache = ROOT/"data/cellsr_gait_antwerp/mat_metadata_blocks"
        self.cache.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.url = "https://api.figshare.com/v2/file/download/42450096"

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [0,self.pos,self.size][whence]+offset
        return self.pos

    def readinto(self, buffer):
        count = min(len(buffer),self.size-self.pos)
        done = 0
        while done < count:
            index,offset = divmod(self.pos,self.block)
            file = self.cache/str(index)
            if not file.exists():
                a = index*self.block
                b = min(a+self.block,self.size)-1
                for attempt in range(5):
                    try:
                        r = self.session.get(self.url,headers={"Range":f"bytes={a}-{b}"},timeout=30)
                        r.raise_for_status()
                        assert r.headers["Content-Range"] == f"bytes {a}-{b}/{self.size}" and len(r.content)==b-a+1
                        file.write_bytes(r.content)
                        self.url = r.url
                        break
                    except (requests.RequestException,AssertionError,KeyError):
                        self.url = "https://api.figshare.com/v2/file/download/42450096"
                        if attempt==4:raise
            data = file.read_bytes()[offset:offset+count-done]
            buffer[done:done+len(data)]=data
            done+=len(data)
            self.pos+=len(data)
        return done


if __name__=="__main__":
    with h5py.File(FigshareReader(),"r") as f:
        refs=f["Sub/sub_char"][()].ravel()
        assert len(refs)==138
        subjects=pd.read_csv(OUT/"sealed_predictions.csv").target.unique()
        values={s:float(f[refs[int(s[4:])-1]]["Weight"][0,0]) for s in subjects}
    weights=pd.Series(values,name="mass").rename_axis("subject")
    original=pd.read_csv(OUT/"antwerp_reference.csv").set_index("subject").mass
    overlap=weights.index.intersection(original.index)
    delta=(weights.loc[overlap]-original.loc[overlap]).abs()
    print("C3D/MAT cross-check (people, max kg difference):", len(delta),delta.max(),flush=True)
    assert (weights.between(25,200)).all() and len(overlap)>=10
    weights.to_csv(OUT/"publisher_mass.csv")
