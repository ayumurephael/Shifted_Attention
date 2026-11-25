# Paired benchmark comparisons

Run the comparison after both systems have completed scoring the same evaluation release. Paths in the JSON specification resolve relative to the specification file. The CLI reports differences as the left system minus the right system. It fixes NumPy's generator seed at 20261126 and uses 10,000 paired cluster resamples.

~~~bash
python -m shifted_attention.comparison --spec comparison.json --output paired-results.json
~~~

The following example targets one pair of completed seed runs. For the paper experiment, list seeds 17, 29 and 43 and provide each seed's corresponding paths under both systems. Use the same synthetic corpus and official benchmark release for both systems. The optional physical-record CSV has columns `split,file,physical_record` and groups all known files derived from the same original observation. Without that mapping each TSB file is its own cluster unless a per-file evaluation record already contains `physical_record`.

~~~json
{
  "seeds": [17],
  "systems": {
    "full": {"17": {"synthetic": "checkpoints/seed-17/synthetic-test.json", "tsb": "results/seed-17/tsb", "rca": {"chamber": "results/seed-17/rca-chamber/events.json"}}},
    "synchronous": {"17": {"synthetic": "synchronous/checkpoints/seed-17/synthetic-test.json", "tsb": "synchronous/results/seed-17/tsb", "rca": {"chamber": "synchronous/results/seed-17/rca-chamber/events.json"}}}
  },
  "contrasts": [{
    "name": "full-minus-sync", "left": "full", "right": "synchronous",
    "analyses": [
      {"name": "root", "kind": "synthetic", "metric": "root_map", "margin": 0},
      {"name": "detector", "kind": "tsb", "split": "M", "metric": "VUS-PR", "margin": -0.01},
      {"name": "point", "kind": "synthetic", "metric": "point_ap", "holm_family": "secondary"},
      {"name": "univariate", "kind": "tsb", "split": "U", "metric": "VUS-PR", "holm_family": "secondary"}
    ]
  }],
  "joint_claims": {"primary": ["full-minus-sync/root", "full-minus-sync/detector"]}
}
~~~

For a real event panel, add an analysis with `kind=rca`, a `panel` key from the system's RCA table, and the exact `dataset`, `truth_unit` and `mode` values in its events.json. Supported metrics are the event metric fields from the evaluator, including AP, MRR, Hit@1 and Recall@3. Separate a given-event protocol from the detected-event protocol. The light tunnel's device-owner-proxy averages windows within each experiment, then experiments equally. Its report includes both scores for each individual experiment and seed.

Synthetic tests export per-window root AP and an adjacent `.ancestor-points.npy` file. The latter holds the observed float32 probability, known count and positive count for each ancestor and score. Multiple rank fragments for an ancestor are additive. Exact pooled point AP is recomputed for each bootstrap sample, preserving ties, by multiplying score-by-ancestor sparse count matrices by the sampled ancestor multiplicities. The CPU path uses SciPy. Passing `--device cuda` uses sparse PyTorch matrices and FP64 reductions on a CUDA GPU. Older test artifacts without ancestor score counts need one rerun of test-set inference to compare pooled point AP.

Each seed's estimate uses paired finite observations. Cluster multiplicities are shared across both arms and all seeds, with descendants, incidents or files kept together. Seed differences are equally averaged inside each resample. The output includes total and paired finite units, individual seed estimates, requested and finite resamples, the uncorrected percentile interval, and TSB source strata. Missing or undefined scores keep their place in coverage accounting. The TSB complete-Eva flag requires 350 paired finite U files or 180 paired finite M files for every seed. Fewer than three observed clusters are treated descriptively in this implementation. An analysis can also set `descriptive=true`. If fewer than 99% of resamples have a defined difference, the interval is absent.

A lower percentile bound above the supplied margin passes that component of the joint comparison. A positive lower bound is required to report an improvement. The optional supplementary p-value tests the chosen margin with a centered paired bootstrap and a plus-one correction. Use `alternative=greater` or `two-sided`. Only explicitly named `holm_family` groups receive Holm adjustment; specify each family and margin before examining the scores. An unavailable test retains its multiplicity slot. Automatically generated source-stratified estimates are descriptive follow-ups to the named contrasts.

## Compute and exposure

The training output contains `parameter-ledger.json` with direct counts for every trainable module and inclusive counts for top-level components. `performance.json` records actual input rows, windows, training throughput, per-device peak memory, training GPU hours and job GPU hours including held-out scoring. Committed counters and the consumption ledger persist across resume. The FLOP estimate counts supervised forward dense matrix operations at the actual padded microbatch shapes, including attention dot products. A multiply-add counts as two operations. Elementwise operations, optimizer work, backward, activation recomputation and optional reconstruction are outside that estimate.

Use the width profiler for comparable batch-one forward/backward measurements on actual generated records. The default runs 20 warmup windows and 200 measured windows for each of the seven widths. It includes both supervised heads, retains the selected configuration, and reports the number of distinct records measured. It performs no parameter updates. End-to-end optimizer throughput is recorded by the training entry point.

~~~bash
python -m shifted_attention.profiling --split /data/sster/test --checkpoint /data/checkpoints/seed-17/best.pt --device cuda --output by-width.json
~~~
