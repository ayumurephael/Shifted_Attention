"""Post-run or sampler-only coverage audit using the generator's private receipts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .data import Corpus, atomic_json, build_index
from .train import id_sequence_digest, prior_indices


QUALITY = (
    "no_visible_root", "only_propagation_visible", "no_time_witness",
    "all_positive_rows_thinned", "new_propagation_onset_after_last_event", "right_censored",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def counts_by_cell(corpus, indices, released):
    """Join a sampled index multiset to ID-checked receipts in shard order."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("expected one or more integer training indices")
    if int(indices.min()) < 0 or int(indices.max()) >= len(corpus):
        raise ValueError("training index outside released manifest")
    multiplicity = np.bincount(indices.astype(np.int64), minlength=len(corpus))
    cells = {(n,k,m): Counter() for n,k,m,_ in released["cells"]}
    template_keys = sorted({tuple(row[:-1]) for field in ("focal_counts","event_counts")
                            for row in released[field]})
    templates = {key: Counter() for key in template_keys}
    axes = {n: np.zeros((2,n), dtype=np.int64) for n,_,_ in cells}
    seen = 0
    for shard_number, name in enumerate(corpus.meta["shards"]):
        path = Path(name)
        if path.parent != Path("shards") or path.suffix != ".tar":
            raise ValueError(f"invalid generator shard name {name}")
        receipt = read_json(corpus.split/"receipts"/path.with_suffix(".json").name)
        rows = receipt["rows"]
        if receipt["public"] != name or not rows or seen+len(rows) > len(corpus):
            raise ValueError(f"receipt does not match manifest shard {name}")
        records = corpus.table[seen:seen+len(rows)]
        if not np.all(records["shard"] == shard_number):
            raise ValueError(f"receipt row count differs from manifest for {name}")
        for offset, row in enumerate(rows):
            if row["id"].encode("ascii") != records["id"][offset]:
                raise ValueError(f"receipt ID differs from indexed manifest at row {seen+offset}")
            visits = int(multiplicity[seen+offset])
            if not visits:
                continue
            key = row["N"], row["K"], row["mode"]
            cell = cells[key]
            cell["windows"] += visits
            cell["direct"] += visits * len(row["raw"])
            cell["known_positive"] += visits * len(row["known"])
            for axis, field in enumerate(("raw","known")):
                axes[key[0]][axis,row[field]] += visits
            quality = row.get("quality",{})
            if "quality_flags" in released and set(QUALITY)-quality.keys():
                raise ValueError(f"missing declared visibility fields in receipt {name}")
            for field in QUALITY:
                if field in quality:
                    cell[field] += visits * int(quality[field])
                    cell[(field,"recorded")] += visits
            if row["focal"]:
                layer, kind, _ = row["focal"]
                ticket = templates[key + (layer,kind)]
                ticket["designated"] += visits
                ticket["designated_witnessed"] += visits * int(row["focal_witness"])
            occurrences, witnessed = set(), set()
            for event in row["events"]:
                # Older receipts have three columns and a window-level target witness.
                layer, kind, visible = event[:3]
                item = key + (layer,kind)
                template = templates[item]
                template["events"] += visits
                template["events_witnessed"] += visits * int(visible)
                occurrences.add(item)
                if visible:
                    witnessed.add(item)
            for item in occurrences:
                templates[item]["windows"] += visits
            for item in witnessed:
                templates[item]["windows_witnessed"] += visits
        seen += len(rows)
    if seen != len(corpus) or sum(c["windows"] for c in cells.values()) != len(indices):
        raise ValueError("receipt join did not cover the complete manifest and selection")
    cell_rows = []
    for (n,k,mode), c in sorted(cells.items()):
        cell_rows.append(dict(N=n,K=k,mode=mode,windows=c["windows"],direct=c["direct"],
                              known_positive=c["known_positive"],quality={
                                  f: dict(count=c[f] if c[(f,"recorded")] else None,
                                          denominator=c[(f,"recorded")],
                                          fraction=c[f]/c[(f,"recorded")] if c[(f,"recorded")] else None)
                                  for f in QUALITY}))
    fields = ("designated","designated_witnessed","events","events_witnessed",
              "windows","windows_witnessed")
    template_rows = [dict(N=n,K=k,mode=m,layer=layer,template=kind,
                          **{f: count[f] for f in fields})
                     for (n,k,m,layer,kind),count in sorted(templates.items())]
    witness_unit = ("target_root_witness_in_window" if "events_with_witnessed_root" in released
                    else "direct_witness_within_event")
    return dict(cells=cell_rows,templates=template_rows,template_witness_definition=witness_unit,
                root_exposure={n:dict(raw=v[0].tolist(),known=v[1].tolist()) for n,v in axes.items()})


def audit(dataset, seed, budget, checkpoint_dir=None):
    split = Path(dataset)/"train"
    build_index(split)
    corpus = Corpus(split)
    release = read_json(split/"audit.json")
    manifest_hash = corpus.meta["manifest_sha256"]
    expected = prior_indices(corpus.table,seed,budget,budget)
    expected_hash = id_sequence_digest(corpus.table,expected)
    result = dict(seed=seed,budget=budget,release_manifest_sha256=manifest_hash,
                  id_sequence_sha256=expected_hash,source="sampler_only")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        complete = read_json(checkpoint_dir/"complete.json")
        snapshot = torch.load(checkpoint_dir/"latest.pt",map_location="cpu",weights_only=False)
        contract = snapshot["contract"]
        if (contract["seed"] != seed or contract["dataset"][0] != manifest_hash
                or contract["steps"]*contract["global_batch"] != budget
                or snapshot["step"] != contract["steps"] or complete["step"] != snapshot["step"]
                or complete["consumed"] != budget or snapshot["cursor"] != budget):
            raise ValueError("completed checkpoint belongs to a different schedule or release")
        actual = snapshot["consumed_indices"]
        if (not np.array_equal(actual,expected) or complete["consumed_id_sha256"] != expected_hash
                or complete["reconstructed_prefix"] != snapshot["reconstructed_prefix"]):
            raise ValueError("recorded consumption differs from sampler or completion marker")
        result.update(source="completed_training",reconstructed_prefix=int(snapshot["reconstructed_prefix"]))
        del snapshot
    result.update(counts_by_cell(corpus,expected,release))
    if checkpoint_dir is not None:
        keys = ("N","K","mode","windows","direct","known_positive")
        measured = sorted(tuple(r[x] for x in keys) for r in read_json(checkpoint_dir/"consumption.json"))
        joined = sorted(tuple(r[x] for x in keys) for r in result["cells"] if r["windows"])
        if joined != measured:
            raise ValueError("receipt exposure differs from directly observed training batches")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--budget",type=int,default=1_440_000)
    p.add_argument("--checkpoint-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.budget <= 0:
        p.error("budget must be positive")
    result = audit(args.dataset,args.seed,args.budget,args.checkpoint_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)


if __name__ == "__main__":
    main()
