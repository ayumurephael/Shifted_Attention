"""Physical time distances of admissible shifted-attention reads on public data."""
import numpy as np


def physical_lags(time, row_valid, token_valid, width, shifts, common_support=False):
    """Return one ledger per batch member from the encoder's actual [B,C,P] token mask.

    Each count is the number of ordered, distinct receiver/source channel pairs
    at the given target patch and offset. Consumers apply the receiving head's
    nonzero coefficient mask. Counts describe available reads, not attention mass.
    """
    time = np.asarray(time, dtype=np.float64)
    rows = np.asarray(row_valid, dtype=bool)
    valid = np.asarray(token_valid, dtype=bool)
    b, t = time.shape
    if width < 1 or t < 1 or rows.shape != time.shape or valid.ndim != 3:
        raise ValueError("invalid clock, row mask, token mask or patch width")
    p = (t+width-1)//width
    if valid.shape[0] != b or valid.shape[2] != p or not np.isfinite(time[rows]).all():
        raise ValueError("token mask shape or real clock disagrees with the input")
    ledgers = []
    for clock, real, observed in zip(time, rows, valid):
        origin = float(clock[real][0]) if real.any() else 0.
        count = np.pad(real,(0,p*width-t)).reshape(p,width).sum(-1)
        relative = np.zeros(t)
        relative[real] = clock[real]-origin
        center = np.pad(relative,(0,p*width-t)).reshape(p,width).sum(-1)/np.maximum(count,1)
        observed = observed & (count>0)[None,:]
        donor_by_shift = []
        for d in shifts:
            ids = np.arange(p)+d
            donor_by_shift.append(observed[:,ids.clip(0,p-1)] & (ids>=0) & (ids<p))
        common = np.logical_and.reduce(donor_by_shift) if common_support else True
        offsets = []
        for d,donor in zip(shifts,donor_by_shift):
            donor = donor & common
            pairs = observed.sum(0)*donor.sum(0)-(observed & donor).sum(0)
            query = np.flatnonzero(pairs)
            source = query+d
            offsets.append(dict(shift=int(d),target_patch=query.tolist(),
                                donor_minus_target=(center[source]-center[query]).tolist(),
                                ordered_channel_pairs=pairs[query].tolist()))
        packed=np.packbits(observed.ravel(order="C"),bitorder="little")
        ledgers.append(dict(origin=origin,rows_per_patch=count.tolist(),
                            token_valid_shape=list(observed.shape),token_valid_little_endian_hex=packed.tobytes().hex(),
                            centers_from_origin=[float(v) if k else None for v,k in zip(center,count)],
                            offsets=offsets))
    return ledgers


def accumulate_lags(ledger, totals):
    """Pool exact pair-weighted moments with bounded memory across probe windows."""
    for r in ledger["offsets"]:
        counts = np.asarray(r["ordered_channel_pairs"],dtype=np.int64)
        if not counts.size:
            continue
        delta = np.asarray(r["donor_minus_target"],dtype=np.float64)
        a = totals.setdefault(r["shift"],dict(count=0,total=0.,minimum=np.inf,maximum=-np.inf))
        a["count"] += int(counts.sum())
        a["total"] += float(delta @ counts)
        a["minimum"] = min(a["minimum"],float(delta.min()))
        a["maximum"] = max(a["maximum"],float(delta.max()))


def head_lags(shifts, coefficients, totals):
    report = []
    for d, w in zip(shifts,coefficients):
        a = totals.get(d) if w != 0 else None
        report.append(dict(shift=int(d),coefficient_active=bool(w!=0),ordered_channel_pairs=a["count"] if a else 0,
                           minimum=a["minimum"] if a else None, maximum=a["maximum"] if a else None,
                           mean=a["total"]/a["count"] if a else None))
    return report
