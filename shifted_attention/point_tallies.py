"""Lossless sufficient statistics for a clustered bootstrap of pooled point AP."""
from pathlib import Path
import numpy as np


TALLY_DTYPE = np.dtype([("ancestor","<u8"),("probability","<f4"),("known","<u4"),("positive","<u4")])


def ancestor_code(width, family, index):
    return (((int(width) << 2) | int(family)) << 32) | int(index)


class AncestorWriter:
    """Keep just one ancestor's points per rank. The reader sums fragments from ranks."""
    def __init__(self, path):
        self.file = Path(path).open("wb")
        self.code = None
        self.scores = []
        self.labels = []

    def add(self, code, scores, labels):
        if code != self.code:
            self.flush()
            self.code = code
        if len(scores):
            self.scores.append(scores)
            self.labels.append(labels)

    def flush(self):
        if self.scores:
            probs,inverse,counts = np.unique(np.concatenate(self.scores),return_inverse=True,return_counts=True)
            records = np.empty(len(probs),dtype=TALLY_DTYPE)
            records["ancestor"] = self.code
            records["probability"] = probs
            records["known"] = counts
            records["positive"] = np.bincount(inverse,weights=np.concatenate(self.labels),minlength=len(probs))
            records.tofile(self.file)
            self.scores.clear()
            self.labels.clear()

    def close(self):
        try:
            self.flush()
        finally:
            self.file.close()


def publish_tallies(parts, path):
    parts = list(parts)
    size = sum(p.stat().st_size for p in parts)
    if size % TALLY_DTYPE.itemsize:
        raise IOError("Truncated point sufficient statistics")
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as dest:
        header = dict(descr=np.lib.format.dtype_to_descr(TALLY_DTYPE),fortran_order=False,
                      shape=(size//TALLY_DTYPE.itemsize,))
        np.lib.format.write_array_header_1_0(dest,header)
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(4*1024**2):
                    dest.write(block)
    tmp.replace(path)
