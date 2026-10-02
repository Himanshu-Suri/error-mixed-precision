# Per-Tile Leading-Zero Quantization of W8A8 Language Models

Measures what leading-zero-derived mixed precision costs in WikiText-2
perplexity on models that are already W8A8.

The scheme: for every K-tile of every quantized tensor, count the leading
zeros of the stored int8 values, reduce those counts to one bit-width per
tile, pair the weight and activation requirements, and promote to the
nearest supported hardware mode. Then apply it and measure the damage.

---

## The scheme in one page

A tile is `k_tile` columns of a weight matrix, spanning all `M` output
rows. Every tile gets one bit-width `b` and one scale shared by all its
values.

**Choosing `b`.** An int8 value with `l` leading zeros needs `8 - l`
bits. Calibration builds a histogram of `l` over each tile, and
`--bit-stat` reduces it to a single number:

| statistic | rule | effect |
|---|---|---|
| `min` | cover every element | lossless, saves almost nothing |
| `percentile` | cover `p`% of elements | interpolates between the two |
| `mean` | track the typical element | aggressive, and the scheme as originally proposed |

This runs separately for weights and activations. The tile takes
`max(w, a)`, promoted up to the nearest member of `--bucket-bits`,
because one multiplier has one precision.

**Applying `b`.** A tile whose largest value needs `m` bits, given only
`b`, has a shortfall of `s = m - b`. That shortfall is the whole error
story, and `--sim-mode` decides what to do about it:

| mode | scale | error | hardware |
|---|---|---|---|
| `shift` | `2^s` | rounding | a shifter |
| `rescale` | `max / (2^(b-1) - 1)` | rounding, finer | a multiplier per tile |
| `truncate` | none | saturation | nothing, and it does not work |

`shift` is the default and the hardware story: a power-of-two scale is a
shift, so the datapath is a narrow multiplier plus a 3-bit exponent
adder plus one shifter per tile. `rescale` is the llama.cpp Q4_0 rule
generalized to any `b`, included as a diagnostic for what the
power-of-two restriction costs. `truncate` is a control that shows why
a shared scale is needed at all.

**What the quantizer does**, in three lines:

```python
reduced = round(value / divisor)     # divisor is 2^s, or the float scale
reduced = clamp(reduced, lo, hi)     # b bits of range
reduced = reduced * divisor          # back onto the original scale
```

Weights go through this once per configuration. Activations go through
it live, every token, every layer, via a hook into
`compressed_tensors`.

**Why the simulation is exact.** Writing `k * 2^s` back into the int8
datapath gives the same product a `b`-bit multiplier would compute:
`(k_w · 2^s_w)(k_a · 2^s_a) = (k_w · k_a) · 2^(s_w + s_a)`. The wider
unit's extra bits are simply idle. So the perplexity measured here is
the perplexity the hardware would produce.

---

## Scripts

Three files. The first does the work; the other two drive it.

### `wxax.py` - the harness

Loads a W8A8 checkpoint, recovers its int8 weights, calibrates, freezes
per-tile bit-widths, applies them, and measures WikiText-2 perplexity
against the untouched baseline.

```bash
python wxax.py \
  --models RedHatAI/Qwen2.5-7B-quantized.w8a8 \
  --k-sizes 16 --arms baseline both \
  --bit-stat mean --sim-mode shift --bucket-bits 2 5 6 8 \
  --trials 8 --calib-seq-len 2048 --eval-seq-len 2048
```

Useful flags beyond the ones above:

| flag | what it does |
|---|---|
| `--arms` | `baseline`, `force8`, `w_only`, `a_only`, `both`. `force8` is the correctness gate: at b=8 the quantizer is an exact pass-through, so a nonzero delta means it is not on the live datapath. `w_only` / `a_only` attribute the damage to each side. |
| `--max-windows N` | Cap the evaluation at N of 146 windows. Smoke tests only. |
| `--diagnose` | Per-tensor SNR, depth-wise hidden-state divergence, logit KL, worst per-token regressions. One extra forward pass per arm. |
| `--fp16-ref MODEL` | Also evaluate the unquantized parent, so the W8A8 step's own cost is visible and the tiling deltas can be read against it. |
| `--pair-mode exact` | Give W and A their own bit-widths instead of `max(w, a)`. A harsher lower bound. |
| `--calib-preset multi` | Five-domain calibration instead of WikiText only. Slower, and the reportable setting. |
| `--dither` | `stochastic` or `error-feedback` rounding. Both roughly double per-element MSE; the benefit flows only through the activations' DC component, which the run reports. |

Writes a timestamped log to `results/`, named for the configuration.

### `census.py` - several models, with a precision census

Runs the harness across a list of models and adds a table the harness
does not print: how much of each model landed in each precision mode,
counted both by tile and by weight parameter.

```bash
python census.py --harness wxax.py \
  --models MODEL_A MODEL_B MODEL_C \
  --bit-stat mean --sim-mode shift --bucket-bits 2 5 6 8
```

`--census-only` stops after calibration, which takes minutes per model
instead of an hour. Use it to check that every checkpoint loads and that
the bit-widths land somewhere sensible before committing to a long run.

**Why two counts.** A tile spans `k_tile × M` parameters, and `M` varies
by an order of magnitude across a model. In Qwen2.5-7B a `k_proj` tile
holds 8,192 parameters and a `down_proj` tile holds 57,344; both count
as one tile. So tile shares say how often the scheme chose a narrow
width, and weight shares say how much of the model is actually stored
narrower. They diverge when the aggressive assignments cluster in narrow
tensors, and the census flags that case.

### `rotate.py` - the same, with a Hadamard rotation

Applies a block-diagonal Hadamard along K before quantization and
removes it after. Everything else comes from the harness unchanged, so
any delta against an unrotated run is attributable to the rotation.

```bash
python rotate.py --harness wxax.py --rot-block 16 --k-sizes 16 \
  --models MODEL_A MODEL_B \
  --bit-stat mean --sim-mode shift --bucket-bits 2 5 6 8
```

**Why rotate.** The error is `2^s` with `s = m - b`, where `b` tracks the
typical element and `m` the largest. Every bit of that gap is waste. A
Hadamard spreads each outlier's energy across its block, so the tile
maximum falls toward the median and `m` drops.

It is free mathematically: for orthonormal `H`,
`Wx = W(HᵀH)x = (WHᵀ)(Hx)`. The dot product is unchanged, and
quantization noise is norm-preserved, so noise reduced in the rotated
basis is reduced in the original one.

**The round trip.** Deployed hardware folds the rotation into adjacent
layers and never un-rotates. Here it must, because the hook returns
integers that `compressed_tensors` then dequantizes with its own scale.
So the simulation is rotate → re-quantize per row → tile-quantize → undo
the rescale → rotate back. Sylvester Hadamard is symmetric and
orthonormal, so `HH = I` and the round trip is exact: only the
quantization noise survives.

**`--no-rotate`** sets the block to 1, the identity. That keeps the
per-row re-quantization and removes the Hadamard, which separates how
much of the gain is each. The two are not equal, so running the control
is worth the hour.

**Expect `eff_bits` to rise.** Rotation destroys the near-zero elements
that were producing high leading-zero counts and therefore low `b`. You
are trading compression for accuracy. Judge the result on perplexity
against bits, not on either alone.

---

## The four configurations

Two independent knobs whether the values are rotated before
quantization, and whether the tile scale is a power of two give four
configurations. Two scripts and one flag cover all of them.

| | no rotation | rotation |
|---|---|---|
| **power-of-two scale** | `census.py --sim-mode shift` | `rotate.py --sim-mode shift` |
| **float scale** | `census.py --sim-mode rescale` | `rotate.py --sim-mode rescale` |

Written out, with everything else held fixed:

```bash
MODELS="RedHatAI/Qwen2.5-7B-quantized.w8a8"
COMMON="--bit-stat mean --bucket-bits 2 5 6 8 \
        --trials 8 --calib-seq-len 2048 --eval-seq-len 2048"

# 1. plain - the scheme as proposed
python census.py --harness wxax.py --models $MODELS \
  --sim-mode shift $COMMON

# 2. llama.cpp float scale - what the power-of-two restriction costs
python census.py --harness wxax.py --models $MODELS \
  --sim-mode rescale $COMMON

# 3. rotation - Hadamard before quantization, shifter kept
python rotate.py --harness wxax.py --rot-block 16 --k-sizes 16 \
  --models $MODELS --sim-mode shift $COMMON

# 4. rotation + float scale - both fixes together
python rotate.py --harness wxax.py --rot-block 16 --k-sizes 16 \
  --models $MODELS --sim-mode rescale $COMMON
```

```bash
# the shift arithmetic is exact when s = 0, so this MUST give dPPL = 0
python wxax.py --models $MODELS --k-sizes 16 --arms baseline both \
  --bit-stat min --sim-mode shift

# identity rotation: keeps the per-row rescale, drops the Hadamard,
# which separates how much of row 3's gain is each
python rotate.py --harness wxax.py --no-rotate --k-sizes 16 \
  --models $MODELS --sim-mode shift $COMMON
```

---

## Reading the output

**`buckets -> b=5: 93.8%, b=6: 6.1%`** - where the tiles landed after
promotion. If one bucket holds nearly everything, the scheme has
collapsed to uniform precision and the adaptivity is not contributing.

**`effective bits = 5.089`** - mean operand width, weighted by the
arithmetic each tile represents. 8.000 is no compression.

**`mean shift depth = 3.193`** - the average of `s = m - b`. This is the
quantity that predicts damage within a model. It does **not** rank
models against each other: architectures differ in how much noise they
absorb.

**`W clip` / `A clip`** - the share of values that fell outside the
representable range.

**`Histogram/running-sum agreement: exact`** - the harness computes the
mean bit-width two independent ways and refuses to continue if they
disagree.

---

## Caveats

**Calibration and evaluation share a corpus.** Bit-widths are derived on
WikiText-2 train and evaluated on WikiText-2 test. Same domain, so the
deltas are mildly optimistic. `--calib-preset multi` is the honest
setting.

**Absolute perplexities are not comparable across models.** Each
tokenizer splits WikiText-2 differently, giving 289k to 356k tokens and
different window counts. Only the relative deltas compare.

**Machine matters.** Baselines reproduce to four decimals within a
machine but can differ slightly across machines. Compare arms measured
in the same session.

**Activation ranges are frozen.** `m` for activations comes from
calibration, so evaluation-time values can exceed it and clip. That is a
real property of any static scheme, not a simulation artifact.

**Attention matmuls are untouched.** The hook covers the quantized
linear layers only, so "100% of compute below 8 bits" refers to those
layers, not the whole model.

**Rescale is a diagnostic, not a design.** A per-tile float scale needs
a real multiplier, which gives back the cheap datapath the scheme exists
to buy. Report it as what the restriction costs.

---

## Prior art

The `shift` mode is microscaling: a shared power-of-two scale per block,
standardized by the Open Compute Project as MXINT/MXFP and shipping in
recent accelerators. The older name is block floating-point.

Applying a Hadamard before block quantization is also established 
QuaRot, SpinQuant, MR-GPTQ.

---

## Environment

```
torch
transformers
datasets
accelerate
compressed-tensors==0.19.0
```

Models are `compressed-tensors` W8A8 checkpoints with per-channel weight
scales and dynamic per-token activation quantization. They load
compressed and decompress on the first forward pass, which is why the
harness runs a warm-up forward before snapshotting the integers.