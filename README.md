# Dual-stream Transformer with Learnable Shifted Multi-head Attention

A PyTorch model returns a marginal root probability for each eligible sensor and a point anomaly probability for each time step. The encoder takes only observed measurements, the observation mask, physical timestamps and the padding masks. It accepts variable channel counts and irregularly spaced retained observations. Inference uses bidirectional offline context.

## Network

The default model has **14,841,458 trainable parameters**. Each eight-row patch follows an affine projection plus a learned GELU correction initialized to zero. Eight blocks maintain independent temporal and relational streams of width 192. Six-head temporal self-attention contextualizes each sensor. Six shifted channel heads read other sensors at patch offsets (0, −1, +1, −2, +2, −4). The receiving head applies the same learned lag coefficients to projected keys and values, masks unsupported lags and gates the attention output before concatenation. The two output streams form a 384-dimensional representation.

The root head applies four learned PMA seeds over each sensor’s patch history, concatenates and compresses the summaries, and uses sensor self-attention before scoring. The point head decodes the eight distinct row positions of each patch, applies two learned PMA seeds to the observed sensors at that row, then compresses and scores. Readout validity is separate from logits.

~~~python
import torch
from shifted_attention.config import ModelConfig
from shifted_attention.model import ShiftedAttentionModel

model = ShiftedAttentionModel(ModelConfig()).eval()
# values [B,T,C] float32, observed [B,T,C] bool, time [B,T] float64.
# channel_valid [B,C] and row_valid [B,T] identify unpadded coordinates.
with torch.inference_mode():
    y = model(values, observed, time, channel_valid, row_valid)
    p_root = torch.where(y["root_valid"], y["root_logits"].sigmoid(), 0.)
    p_point = torch.where(y["anomaly_valid"], y["anomaly_logits"].sigmoid(), 0.)
~~~

## Train on the published corpus

Use Python 3.10 or later and a CUDA-enabled PyTorch build for H100 training. The corpus must contain three complete splits from the paired Data-Generator repository. Each split contains a manifest and uncompressed tar shards with separately indexed input and target members. The loader indexes manifest offsets on disk, opens shard handles in each worker and only sends the public measurement fields to the network. Keep the split mounted read-write during first-time reader index construction. [Data-Generator](https://github.com/ayumurephael/Data-Generator)

The first-round release `round1-1680k-v1` has 1,680,000 training windows, with 240,000 at each sensor width. It uses the original validation and test splits of 336,000 windows each and their calibration scale bank. The training audit and success marker identify this edition. The original 6,720,000-window training corpus also remains supported. The three manifest hashes identify the exact split contents for checkpoint resume. Use a separate output directory for each edition and seed.

~~~bash
python -m pip install -e '.[test,benchmarks,industrial]'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
for seed in 17 29 43; do
  torchrun --standalone --nproc_per_node=6 -m shifted_attention.train \
    --dataset /data/sster --output "/data/models/seed-${seed}" --seed "$seed" \
    --global-batch 24 --microbatch 4 --workers 2 --resume
done
~~~

Each seed schedules 1,440,000 windows over 60,000 optimizer updates. The sampler uses exact largest-remainder apportionment by sensor width, root count, mode and designated template before shuffling. Six replicas each process four windows per update. The denominators of both BCE losses count valid supervision over every microbatch on every replica. Optimization uses AdamW, a 3,000-step warmup, cosine decay, unit gradient clipping and BF16 CUDA matrix operations. The implementation preserves FP32 master weights. Step checkpoints record model and optimizer, RNG states, consumed windows, sampler position, run configuration and manifest hashes.

Validation partitions whole ancestors within each width and mechanism family. Every 2,000 steps the selection score averages nondegenerate event-wise root AP and pooled point AP. Ties retain the earlier checkpoint. The independent calibration ancestors set a 95th-percentile healthy-window alert threshold. Each seed produces step snapshots, best.pt, latest.pt, calibration.json, synthetic-test.json, consumption.json and complete.json. Fixture mode explicitly selects a smaller verification configuration.

## Audit training coverage

Under BF16 autocast, linear products use BF16 operands and native autograd. Biases, position additions, residual streams, nonlinear updates and channel gates are carried in FP32. The parameter structure and full-precision evaluation path are shared across training and inference.

To audit the frozen sampler before training, run `python -m shifted_attention.consumption --dataset /data/sster --seed 17 --budget 1440000 --output sampler-17.json`. After training, add `--checkpoint-dir /path/to/seed-17` to verify the actual ID order and batch counts. This offline audit joins the selection to private generator receipts and reports direct and witnessed root exposure per channel, each template's scheduled and witnessed counts, repeat occurrences and six visibility flags with per-cell denominators. The optimizer and model read the public inputs and supervision.

The report records the planned prefix and the total visits separately. A global batch with both losses masked advances the sampler and requires another draw to reach the optimizer-update budget. Its visits count toward the realized training exposure. Legacy receipts identify window-level target witnesses and list the denominator for each available visibility flag.

## Evaluate real measurements

The public-data preparation command downloads the released files and verifies the metric source files against their Git object hashes. Pass a local benchmark root with room for the extracted archives. The external tests consume the selected checkpoint. For root localization the optional threshold comes from the same seed’s calibration.json. [Dataset inventory, ranking definitions and release-specific import examples](BENCHMARKS.md)

~~~bash
python -m shifted_attention.evaluation prepare --data /data/benchmarks --assets tsad chamber msds
python -m shifted_attention.evaluation timercd --checkpoint /data/models/seed-17/best.pt --data /data/benchmarks --output /data/results/time-rcd
python -m shifted_attention.evaluation tsb --checkpoint /data/models/seed-17/best.pt --data /data/benchmarks --output /data/results/tsb
python -m shifted_attention.evaluation rca --checkpoint /data/models/seed-17/best.pt --data /data/benchmarks --ledger /data/benchmarks/rca/chamber.json --output /data/results/chamber
~~~

The TimeRCD command enforces the paper’s counts across 11 univariate and five multivariate sources and scores the post-reference test suffix. The separate TSB-AD command scores the official Eva lists of 350 univariate and 180 multivariate series. RCA ledgers carry truth units, direct-target or proxy evidence, candidate sets and event bounds. The industrial importer accepts a release-specific spec for locally available SWaT or WADI files. [TimeRCD](https://proceedings.mlr.press/v306/lan26b.html) · [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) · [AERCA](https://github.com/hanxiao0607/AERCA)

## Controlled comparisons

Training accepts the synchronous, fixed-shift, gate-bypass, feedback and flat joint-attention variants through --variant. Additional switches expose a common lag-support mask, linear tokenizer, repeated patch score features, seed counts, patch width and shift range. Use --aux-reconstruction for the independent 15% masked-patch comparison. It adds an observed-point MSE from a second encoder pass, records the extra forward time, and removes the reconstruction head for deployed inference. Keep the same seed, corpus release and consumption budget for paired runs. [Paired reporting](COMPARISONS.md)

## Tests

`python -m shifted_attention.precision --device cuda --draws 8 --output precision.json` compares identical weights and inputs in FP64, FP32 and BF16. Pass `--checkpoint /path/to/best.pt` to use trained weights. It reports absolute error distributions for the shifted operator, raw logits and probabilities, plus valid-mask mismatches. The mechanism probe records physical donor-minus-target time together with patch offsets, admissible ordered sensor-pair counts and the masks used at each lag. [Mechanism probe commands](COMPARISONS.md)

~~~bash
python -m pytest
python -m sster_data smoke --out /tmp/sster-verification --split all --workers 2
SSTER_INTEGRATION_DATA=/tmp/sster-verification python -m pytest tests/test_training_integration.py
~~~

The generator smoke command requires the Data-Generator package. The integration tests exercise the same loading, gradient synchronization, checkpoint selection and recovery paths as production.
