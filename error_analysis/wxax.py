"""
============================================================
WxA ERROR EVALUATION  (mean-LZ + matched-bucket quantization)
============================================================

Companion to the WxA *analysis* script. That script only COUNTED
what bit-width each K-tile would need. This script actually
APPLIES those bit-widths to the same natively-W8A8 model and
measures the resulting WikiText-2 perplexity degradation.

WHAT "QUANTIZE A TILE TO b BITS" MEANS HERE
-------------------------------------------
The model is already W8A8. Mean LZ over a tile says: on average,
the elements of this tile have (8 - b) leading zeros, so they fit
in b bits AT THE EXISTING SCALE. We therefore do NOT rescale
anything -- the per-channel weight scale and the per-token
activation scale are left completely untouched, so dequant is
bit-identical to the native path. Quantizing to b bits is simply:

    Q_b(q) = clamp(q, -(2^(b-1) - 1), +(2^(b-1) - 1))

Consequences worth internalizing before reading any PPL number:

  * The ONLY error mechanism is SATURATION of the above-average
    elements of each tile. There is no rounding error and no LSB
    loss, because we keep the low b bits exactly.
  * Therefore the diagnostic that explains the PPL is the CLIP
    RATE, not any notion of quantization noise. This script logs
    clip rate per arm and per tensor.
  * Larger k_tile should be monotonically worse: one outlier
    channel drags down the mean-LZ bound for the entire tile.

BUCKETING
---------
map_to_bucket_bits((w, a)) = smallest supported b >= max(w, a).
So within a tile the WEIGHT and the ACTIVATION receive the SAME
b. This is strictly more generous than applying w_bits to W and
a_bits to A independently, and it is the actual hardware story
(a fixed set of matched precision modes), so it is the headline
configuration. The independent-pair variant is available via
--pair-mode exact as a lower bound.

CALIBRATION HYGIENE
-------------------
Bit-widths are derived ONCE on the multi-domain calibration set
and then FROZEN. They are never re-derived on WikiText. Doing
otherwise would make the scheme oracle-adaptive and the reported
PPL meaningless.

ARMS
----
  baseline   native W8A8, nothing touched
  force8     every tile forced to b=8  -> MUST match baseline to
             ~1e-4, since clamp(q, -127, 127) is a no-op on int8.
             This is the correctness check for the whole harness.
  w_only     bucket bits applied to weights only, A left at 8
  a_only     bucket bits applied to activations only, W left at 8
  both       the actual proposal
============================================================
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import compressed_tensors
import compressed_tensors.quantization.lifecycle.forward as ct_forward
import compressed_tensors.quantization.lifecycle.forward_helpers as ct_helpers

# ============================================================
# TRANSFORMERS VERSION SHIM
#
# transformers renamed from_pretrained's `torch_dtype` argument to
# `dtype` (v4.56 deprecated it, v5 removed it). Rather than pin a
# version, probe the signature once and use whichever this install
# accepts, so the harness runs on both 4.x and 5.x.
# ============================================================

def _dtype_kwarg(dtype):
    try:
        import inspect
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            return {"dtype": dtype}
    except (TypeError, ValueError):
        pass
    return {"torch_dtype": dtype}


def load_causal_lm(model_name, dtype="auto"):
    """from_pretrained with device_map='auto', dtype arg name auto-detected."""
    kw = dict(device_map="auto", trust_remote_code=True)
    kw.update(_dtype_kwarg(dtype))
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kw)
    except TypeError as e:
        # Signature probe can be defeated by **kwargs-only wrappers;
        # fall back to the other spelling before giving up.
        if "dtype" not in str(e):
            raise
        alt = dict(device_map="auto", trust_remote_code=True)
        if "dtype" in kw:
            alt["torch_dtype"] = dtype
        else:
            alt["dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(model_name, **alt)


_CT_VERSION = getattr(compressed_tensors, "__version__", "unknown")
for _fn in ("forward_quantize",):
    if not hasattr(ct_forward, _fn):
        raise RuntimeError(
            f"compressed_tensors {_CT_VERSION} has no ct_forward.{_fn}; this harness "
            f"patches library internals and needs updating for this version."
        )
if not hasattr(ct_helpers, "round_to_quantized_type_args"):
    raise RuntimeError(
        f"compressed_tensors {_CT_VERSION} has no round_to_quantized_type_args in "
        f"forward_helpers; this harness patches library internals and needs updating."
    )


# ============================================================
# GLOBAL HOOK STATE
#
# One patch serves two jobs:
#   MODE["capture"] -> record native int8 activations (calibration)
#   MODE["apply"]   -> clamp activations to their frozen tile bounds
# They are never both on at once.
# ============================================================

# ---- Streaming calibration statistics ----
# We never retain activation tensors. Mean LZ over a tile is a LINEAR
# statistic, so a running per-column sum of leading-zero counts plus a
# row count is mathematically sufficient:
#
#   mean_LZ(tile t) = sum_{c in t} colsum_LZ[c] / (n_rows * k_tile)
#
# This is EXACTLY equal to pooling every trial into one [N, K] tensor
# and taking the mean -- every column has the same row count -- but it
# costs one float64 [K] vector per module instead of gigabytes of
# retained int8. It also means a single capture pass serves every
# k_tile at once, rather than re-pooling per tile size.
ACT_LZ_COLSUM = {}   # module_id -> float64 [K] on device
ACT_ROWS = {}        # module_id -> int
ACT_ABSMAX = {}      # module_id -> int16 [K] per-column max |q| over calibration
ACT_NEG128 = {}      # module_id -> int, count of exact -128 values seen

# Per-module activation error, accumulated during apply mode:
#   [sum (q'-q)^2, sum q^2, max |q'-q|]
# Ratio of the first two gives a signal-to-quantization-noise ratio per
# tensor, which localizes error far better than clip rate alone: a tensor
# can clip rarely but catastrophically, or often but harmlessly.
ACT_ERR = defaultdict(lambda: [0.0, 0.0, 0.0])

# Per-column SUM of activation values (not magnitudes). Divided by the
# row count this gives each channel's DC level, which is what decides
# whether dithering can help at all: the output error carries a term
# Sum(w'-w) * x_bar, so error-feedback dithering (which drives
# Sum(w'-w) -> 0) only pays off where x_bar is large. On zero-mean
# channels every dither mode is strictly harmful, because it doubles
# per-element MSE and there is no bias term to cancel.
ACT_VALSUM = {}

# Per-column HISTOGRAM of leading-zero counts, shape [K, 8].
#
# LZ takes only 8 values (0..7), so a full per-column histogram costs
# K*8*4 bytes -- about 36 MB for this whole model -- and is independent
# of trial count and sequence length. Retaining the activations
# themselves would cost 4.6 GB at 8x512 and 74 GB at 32x2048.
#
# The histogram is strictly more informative than the running sums: it
# yields the mean (sum(l*h)/n) and the min (lowest occupied bin)
# exactly, and unlike a running sum it also yields any PERCENTILE --
# which is what a bit-selection statistic between "mean" (too
# aggressive) and "min" (lossless but no compression) needs.
ACT_LZ_HIST = {}

LZ_BINS = 8

current_context = {"module_id": None, "layer_name": None, "base_name": None}

MODE = {"capture": False, "apply": False}

# Monkey-patching compressed_tensors internals is version-fragile: a
# rename in forward.py / forward_helpers.py would silently disable the
# activation side rather than erroring. PATCH_HITS lets the harness
# assert the patch actually fired.
PATCH_HITS = {"forward_quantize": 0, "round": 0}

# module_id -> int8 [K] tensor of per-column activation bounds
ACT_BOUNDS = {}

# module_id -> [n_clipped, n_total]  (activation clip accounting)
ACT_CLIP_STATS = defaultdict(lambda: [0, 0])

orig_forward_quantize = ct_forward.forward_quantize
orig_round_helpers = ct_helpers.round_to_quantized_type_args
orig_round_forward = getattr(ct_forward, "round_to_quantized_type_args", None)


def wrapped_forward_quantize(module, value, base_name, args):
    prev = (current_context["module_id"], current_context["layer_name"], current_context["base_name"])
    current_context["module_id"] = id(module)
    current_context["layer_name"] = getattr(module, "_lz_layer_name", None)
    current_context["base_name"] = base_name
    PATCH_HITS["forward_quantize"] += 1
    try:
        return orig_forward_quantize(module, value, base_name, args)
    finally:
        (current_context["module_id"],
         current_context["layer_name"],
         current_context["base_name"]) = prev


def _is_a8_dynamic_token(args):
    return (
        getattr(args, "num_bits", None) == 8
        and str(getattr(args, "type", "")).lower().endswith("int")
        and str(getattr(args, "strategy", "")).lower().endswith("token")
        and bool(getattr(args, "dynamic", False))
    )


def wrapped_round_to_quantized_type_args(tensor, args, *pos_args, **kwargs):
    """
    compressed_tensors hands us the INTEGER-valued tensor here, before
    dequant. That is exactly where the clamp belongs: clamping the
    integers and letting the untouched scale dequant them is precisely
    "this tile only had b bits of range".
    """
    result = orig_round_helpers(tensor, args, *pos_args, **kwargs)
    PATCH_HITS["round"] += 1

    if not (MODE["capture"] or MODE["apply"]):
        return result
    if not _is_a8_dynamic_token(args):
        return result

    module_id = current_context["module_id"]
    if module_id is None or current_context["base_name"] != "input":
        return result
    if not isinstance(result, torch.Tensor) or result.numel() == 0:
        return result

    with torch.no_grad():
        if MODE["capture"]:
            q = result.detach()
            rows = q.reshape(-1, q.shape[-1])

            # Integer-validity check on the first trial only; it costs a
            # full elementwise comparison and the invariant does not vary
            # across trials.
            if module_id not in ACT_ROWS:
                if not torch.equal(rows.float(), torch.round(rows.float())):
                    raise RuntimeError("Native A8 result is not integer-valued.")
                lo, hi = rows.min().item(), rows.max().item()
                if lo < -128 or hi > 127:
                    raise RuntimeError(f"Native A8 result outside INT8 range: [{lo}, {hi}]")

            lz = compute_elementwise_lz(rows.to(torch.int16))
            colsum = lz.sum(dim=0, dtype=torch.float64)
            # Per-column max |q| is what min-LZ (lossless) bit selection
            # and per-tile shift/scale derivation both need. -128 is
            # counted separately so the force8 diagnostic can tell a
            # genuine datapath failure apart from the -128 edge case.
            colmax = rows.abs().to(torch.int16).amax(dim=0)
            colsum_val = rows.sum(dim=0, dtype=torch.float64)
            hist = torch.stack(
                [(lz == b).sum(dim=0, dtype=torch.int64) for b in range(LZ_BINS)],
                dim=1,
            )  # [K, 8]
            n_neg128 = int((rows == -128).sum().item())

            prev = ACT_LZ_COLSUM.get(module_id)
            if prev is None:
                ACT_LZ_COLSUM[module_id] = colsum
                ACT_ROWS[module_id] = rows.shape[0]
                ACT_ABSMAX[module_id] = colmax
                ACT_VALSUM[module_id] = colsum_val
                ACT_LZ_HIST[module_id] = hist
                ACT_NEG128[module_id] = n_neg128
            else:
                if prev.numel() != colsum.numel():
                    raise RuntimeError(
                        f"Activation K changed across trials for module {module_id}: "
                        f"{prev.numel()} -> {colsum.numel()}"
                    )
                prev += colsum
                ACT_ROWS[module_id] += rows.shape[0]
                ACT_ABSMAX[module_id] = torch.maximum(ACT_ABSMAX[module_id], colmax)
                ACT_VALSUM[module_id] += colsum_val
                ACT_LZ_HIST[module_id] += hist
                ACT_NEG128[module_id] += n_neg128

            del lz, colsum, colmax, colsum_val, hist
            return result

        # ---- apply mode ----
        params = ACT_BOUNDS.get(module_id)
        if params is None:
            # No frozen quantizer for this tensor (skipped during
            # calibration, or every tile is already at b=8). Leave it at
            # full int8.
            return result

        if result.shape[-1] != params["lo"].numel():
            raise RuntimeError(
                f"Activation K={result.shape[-1]} does not match frozen quantizer "
                f"K={params['lo'].numel()} for module id {module_id}."
            )

        out = apply_tile_quant(
            result, params, count_clip=ACT_CLIP_STATS[module_id],
            dither=DITHER_CFG["activations"], k_tile=DITHER_CFG["k_tile"],
        )

        e = ACT_ERR[module_id]
        diff = (out.to(torch.float32) - result.to(torch.float32))
        e[0] += float((diff ** 2).sum().item())
        e[1] += float((result.to(torch.float32) ** 2).sum().item())
        e[2] = max(e[2], float(diff.abs().max().item()))
        del diff

        return out


@contextmanager
def patched_quant_path():
    ct_forward.forward_quantize = wrapped_forward_quantize
    ct_helpers.round_to_quantized_type_args = wrapped_round_to_quantized_type_args
    if orig_round_forward is not None:
        ct_forward.round_to_quantized_type_args = wrapped_round_to_quantized_type_args
    try:
        yield
    finally:
        ct_forward.forward_quantize = orig_forward_quantize
        ct_helpers.round_to_quantized_type_args = orig_round_helpers
        if orig_round_forward is not None:
            ct_forward.round_to_quantized_type_args = orig_round_forward


# ============================================================
# LZ / BIT-WIDTH DERIVATION  (identical semantics to the
# analysis script -- kept here so this file runs standalone)
# ============================================================

def compute_elementwise_lz(tensor_2d):
    int_tensor = tensor_2d.to(torch.int32)
    magnitude = torch.abs(int_tensor)

    is_min_val = int_tensor == -128
    magnitude = torch.where(is_min_val, torch.full_like(magnitude, 127), magnitude)

    is_zero = magnitude == 0
    safe_val = torch.where(is_zero, torch.ones_like(magnitude), magnitude)

    magnitude_bits = torch.floor(torch.log2(safe_val.float())).to(torch.int32) + 1
    magnitude_bits = torch.where(is_zero, torch.zeros_like(magnitude_bits), magnitude_bits)

    lz = 7 - magnitude_bits
    return torch.where(is_min_val, torch.zeros_like(lz), lz)


def lz_colsum(tensor_2d):
    """Per-column sum of leading-zero counts, plus the row count."""
    lz = compute_elementwise_lz(tensor_2d)
    return lz.sum(dim=0, dtype=torch.float64), tensor_2d.shape[0]


def bits_from_lz_colsum(colsum, n_rows, k_tile, minimum_bits=2, maximum_bits=8):
    """
    Derive per-tile bit-widths from accumulated per-column LZ sums.
    Identical in value to averaging LZ over a fully materialized [N, K]
    tensor, since every column contributes the same n_rows.
    """
    K = colsum.numel()
    if K % k_tile != 0:
        raise ValueError(f"K={K} not divisible by k_tile={k_tile}")

    tile_sums = colsum.reshape(K // k_tile, k_tile).sum(dim=1)
    mean_lz = tile_sums / (float(n_rows) * float(k_tile))
    bits = torch.clamp(torch.round(8.0 - mean_lz), minimum_bits, maximum_bits)
    return [int(b) for b in bits.tolist()]


# The supported matched-precision modes. 3, 4 and 7 are deliberately
# absent: these are the modes the target hardware actually implements,
# so any tile needing 3 or 4 bits is promoted to 5. That promotion makes
# the tile distribution look better than a [2,4,6,8] set would, and is
# worth stating explicitly when reporting results.
# The matched-precision modes the target hardware implements. Every
# tile's (w, a) pair is promoted UPWARD to the smallest member that is
# >= max(w, a), so this set is a third experimental axis alongside
# --bit-stat and --sim-mode:
#
#   coarser set (e.g. [6, 8])          -> more promotion, better PPL,
#                                         less compression
#   finer set (e.g. [2,3,4,5,6,7,8])   -> less promotion, worse PPL,
#                                         more compression
#
# It is the cheapest axis to sweep: the frozen per-tile bit vectors do
# not change, only how they are mapped, so no recalibration is needed.
#
# Note that a set containing a member exactly where the bit statistic
# piles up is the WORST case, because nothing gets promoted. With
# mean-LZ on this model almost every tile wants 5 bits, so a set
# containing 5 leaves them all at 5; dropping 5 pushes them to 6, which
# halves the shift depth and cuts the rounding step by 2x.
SUPPORTED_BUCKET_BITS = [2, 5, 6, 8]


def set_bucket_bits(bits):
    """Replace the supported matched-precision modes (sorted, deduped)."""
    global SUPPORTED_BUCKET_BITS, BUCKET_LABELS
    vals = sorted({int(b) for b in bits})
    if not vals:
        raise ValueError("bucket set cannot be empty")
    if vals[-1] != 8:
        raise ValueError(
            f"bucket set must contain 8 (the native width) as its largest member; got {vals}"
        )
    if vals[0] < 2:
        raise ValueError(f"bucket bits must be >= 2; got {vals}")
    SUPPORTED_BUCKET_BITS = vals
    BUCKET_LABELS = [f"W{b}A{b}" for b in vals]
    return vals


def map_to_bucket_bits(w_bits_tile, a_bits_tile):
    m = max(w_bits_tile, a_bits_tile)
    for b in SUPPORTED_BUCKET_BITS:
        if b >= m:
            return b
    return SUPPORTED_BUCKET_BITS[-1]


def lz_hist(tensor_2d):
    """Per-column LZ histogram [K, 8] for an integer tensor [N, K]."""
    lz = compute_elementwise_lz(tensor_2d)
    return torch.stack(
        [(lz == b).sum(dim=0, dtype=torch.int64) for b in range(LZ_BINS)], dim=1
    )


def bits_from_lz_hist(hist, k_tile, stat="mean", pct=99.0,
                      minimum_bits=2, maximum_bits=8):
    """
    Per-tile bit-width from a per-column LZ histogram.

    An element with leading-zero count l needs (8 - l) bits, so small l
    means an expensive element. The three statistics differ only in how
    far into the expensive tail they insist on covering:

      min         cover every element -> b = 8 - min(l present).
                  Lossless under truncate mode; the zero-error anchor.
      mean        b = 8 - mean(l). Covers roughly the typical element
                  and ignores the tile's dynamic range entirely, which
                  is why it needs a shift and why it degrades so hard.
      percentile  cover pct% of elements -> take the (100-pct) quantile
                  of l from the LOW end (the expensive tail) so that
                  only (100-pct)% of elements fall outside b bits.
                  Interpolates between the two extremes above.
    """
    K = hist.shape[0]
    if K % k_tile != 0:
        raise ValueError(f"K={K} not divisible by k_tile={k_tile}")

    n_tiles = K // k_tile
    # Pool the histogram over each tile's columns: [n_tiles, 8]
    th = hist.reshape(n_tiles, k_tile, LZ_BINS).sum(dim=1).to(torch.float64)
    totals = th.sum(dim=1).clamp(min=1.0)

    if stat == "mean":
        levels = torch.arange(LZ_BINS, dtype=torch.float64).unsqueeze(0)
        stat_lz = (th * levels).sum(dim=1) / totals
    elif stat == "min":
        occupied = th > 0
        idx = torch.arange(LZ_BINS).unsqueeze(0).expand_as(occupied)
        stat_lz = torch.where(occupied, idx, torch.full_like(idx, LZ_BINS - 1)) \
                       .amin(dim=1).to(torch.float64)
    elif stat == "percentile":
        # Walk the LZ axis from l=0 (most expensive) upward and stop once
        # the cumulative share of elements left BELOW the cut is at most
        # (100 - pct)%, i.e. at most that fraction is sacrificed.
        allowed = (100.0 - float(pct)) / 100.0
        cum = th.cumsum(dim=1) / totals.unsqueeze(1)
        # smallest l whose cumulative-below share already exceeds the
        # allowance; everything strictly below l is what we give up.
        over = cum > allowed
        first = torch.where(
            over.any(dim=1), over.float().argmax(dim=1), torch.zeros(n_tiles, dtype=torch.long)
        )
        stat_lz = first.to(torch.float64)
    else:
        raise ValueError(f"Unknown bit statistic: {stat}")

    bits = torch.clamp(torch.round(8.0 - stat_lz), minimum_bits, maximum_bits)
    return [int(b) for b in bits.tolist()]


def bits_from_absmax(absmax, k_tile, minimum_bits=2, maximum_bits=8):
    """
    Per-tile bit-width from the tile's LARGEST magnitude (equivalently,
    the MINIMUM leading-zero count over the tile). This is the lossless
    choice: a tile assigned b bits this way fits entirely in b bits at
    the native scale, so `truncate` mode is exactly a no-op.
    """
    K = absmax.numel()
    if K % k_tile != 0:
        raise ValueError(f"K={K} not divisible by k_tile={k_tile}")
    tile_max = absmax.reshape(K // k_tile, k_tile).amax(dim=1)
    need = magnitude_bits(tile_max) + 1  # +1 for the sign bit
    return [int(b) for b in torch.clamp(need, minimum_bits, maximum_bits).tolist()]


def magnitude_bits(v):
    """Bits needed for the magnitude |v| (0 -> 0 bits). int tensor in, int tensor out."""
    m = v.abs().to(torch.int32)
    z = m == 0
    s = torch.where(z, torch.ones_like(m), m)
    mb = torch.floor(torch.log2(s.float())).to(torch.int32) + 1
    return torch.where(z, torch.zeros_like(mb), mb)


# ============================================================
# TILE QUANTIZER
#
# A tile assigned b bits can be simulated three ways, and the choice
# changes the answer by more than any other knob in this harness.
# They are NOT interchangeable and correspond to different hardware.
#
#   truncate  Keep the low b bits at the native scale:
#                 q' = clamp(q, -(2^(b-1)-1), +(2^(b-1)-1))
#             Shift is implicitly 0, so this is only a faithful b-bit
#             mode when b >= (bits needed by the tile's largest
#             element). That holds for --bit-stat min and FAILS for
#             --bit-stat mean, where it destroys the tile's high end
#             instead of trading away low-end resolution. Error is
#             pure saturation.
#
#   shift     Per-tile power-of-two scale. The tile's top bit sits at
#             position m-1; a b-bit window starting at bit (m-b) is
#             selected:
#                 s  = max(0, m - b)
#                 q' = clamp(round(q / 2^s), +-(2^(b-1)-1)) * 2^s
#             Exactly representable in int8, so weights can be
#             simulated with zero extra approximation. Hardware cost is
#             a shifter plus ~3 bits of per-tile metadata. Error is
#             rounding, plus residual clipping only where an eval value
#             exceeds the calibration max.
#
#   rescale   Per-tile affine scale s = max|q| / (2^(b-1)-1), i.e. the
#             most generous b-bit mode possible. Needs a real per-tile
#             multiplier in hardware. For activations this is simulated
#             exactly (the datapath is float after this point); for
#             weights the result must be rounded back into int8, which
#             adds up to half an int8 LSB on top -- negligible relative
#             to s, but noted for honesty.
#
# For ACTIVATIONS, m is taken from the calibration set and held STATIC.
# Deriving it from the eval batch would be oracle-adaptive. Values that
# exceed the calibration range at eval time clip, and that clipping is
# counted and reported -- it is a real cost of a static scheme.
# ============================================================

SIM_MODES = ["truncate", "shift", "rescale"]


def build_tile_quant(bits, k_tile, absmax, mode):
    """
    bits:   per-tile bit-widths (list, length K/k_tile)
    absmax: int16 [K] per-column max |q| (from calibration for
            activations, computed offline for weights)
    Returns a dict of expanded per-column [K] tensors, or None if this
    tensor needs no modification at all.

    A tile at b=8 is left completely untouched -- no bound is
    manufactured for it. This makes the force8 arm a no-op BY
    CONSTRUCTION rather than by luck, and in particular stops -128 from
    being silently clamped to -127 (the native int8 range is asymmetric;
    the symmetric +-(2^(b-1)-1) formula is a deliberate choice for b<8
    only, where a lopsided representable range would be worse).
    """
    b = torch.tensor(bits, dtype=torch.int32)
    n_tiles = b.numel()
    if absmax.numel() != n_tiles * k_tile:
        raise ValueError(f"absmax K={absmax.numel()} != n_tiles*k_tile={n_tiles * k_tile}")

    if bool((b >= 8).all()):
        return None  # nothing to do: full int8 everywhere

    is_full = b >= 8
    sym = 2 ** (b - 1) - 1

    # b=8 tiles get the true native int8 range, not the symmetric one.
    hi = torch.where(is_full, torch.full_like(sym, 127), sym)
    lo = torch.where(is_full, torch.full_like(sym, -128), -sym)

    params = {
        "lo": lo.repeat_interleave(k_tile).to(torch.int16),
        "hi": hi.repeat_interleave(k_tile).to(torch.int16),
        "shift": None,
        "scale": None,
    }

    if mode == "truncate":
        return params

    tile_max = absmax.reshape(n_tiles, k_tile).amax(dim=1)
    m = magnitude_bits(tile_max) + 1  # bits the tile's largest element needs

    if mode == "shift":
        s = torch.clamp(m - b, min=0)
        s = torch.where(is_full, torch.zeros_like(s), s)
        params["shift"] = s.repeat_interleave(k_tile).to(torch.int16)
        return params

    if mode == "rescale":
        denom = sym.clamp(min=1).to(torch.float32)
        sc = tile_max.to(torch.float32) / denom
        sc = torch.clamp(sc, min=1.0)              # never expand range
        sc = torch.where(is_full, torch.ones_like(sc), sc)
        params["scale"] = sc.repeat_interleave(k_tile).to(torch.float32)
        return params

    raise ValueError(f"Unknown sim mode: {mode}")


DITHER_MODES = ["none", "stochastic", "error-feedback"]

_DITHER_GEN = {"gen": None}

# Which side gets dithered, and the tile size error-feedback runs over.
# Set per arm before any quantization happens.
DITHER_CFG = {"weights": "none", "activations": "none", "k_tile": None}


def set_dither_seed(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    _DITHER_GEN["gen"] = g


def _round_dithered(x, mode, k_tile):
    """
    Rounding step with optional dithering. Only reached when a divisor is
    in play (shift / rescale modes) -- truncate has no rounding step, so
    dithering is a genuine no-op there.

    Measured behaviour (see notes in the module docstring): both modes
    roughly DOUBLE per-element MSE relative to round-to-nearest. That
    cost is unconditional. The benefit is conditional and flows only
    through the activations' DC component, via the output-error term
    Sum(w'-w) * x_bar:

      stochastic      makes each error zero-mean and independent. Cuts
                      the bias term but never beats round-to-nearest on
                      dot-product RMSE at any DC level. Treat as control.
      error-feedback  carries the rounding residual forward along K
                      inside each tile, so the tile's SIGNED error sums
                      to <= 0.5 LSB by construction. Loses to
                      round-to-nearest on zero-mean channels, wins
                      substantially once channel means are large -- i.e.
                      exactly the outlier-channel regime.
    """
    if mode == "none":
        return torch.round(x)

    if mode == "stochastic":
        gen = _DITHER_GEN["gen"]
        if gen is not None and gen.device == x.device:
            u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        else:
            u = torch.rand(x.shape, device=x.device, dtype=x.dtype)
        return torch.floor(x + u)

    if mode == "error-feedback":
        K = x.shape[-1]
        if K % k_tile != 0:
            return torch.round(x)
        flat = x.reshape(-1, K // k_tile, k_tile)
        out = torch.empty_like(flat)
        resid = torch.zeros(flat.shape[:-1], device=x.device, dtype=x.dtype)
        for j in range(k_tile):
            v = flat[..., j] + resid
            r = torch.round(v)
            resid = v - r
            out[..., j] = r
        return out.reshape(x.shape)

    raise ValueError(f"Unknown dither mode: {mode}")


def apply_tile_quant(q, params, count_clip=None, dither="none", k_tile=None):
    """
    Apply a tile quantizer to an integer-valued tensor whose last
    dimension is K. Works for weights [M, K] and activations [..., K].
    Returns a tensor of the same dtype.

    count_clip: optional 2-element list, incremented as [n_clipped, n_total].
    """
    if params is None:
        return q

    dev = q.device
    lo = params["lo"].to(dev)
    hi = params["hi"].to(dev)

    work = q.to(torch.float32)

    shift = params["shift"]
    scale = params["scale"]
    if shift is not None:
        divisor = torch.pow(2.0, shift.to(dev).to(torch.float32))
    elif scale is not None:
        divisor = scale.to(dev)
    else:
        divisor = None

    if divisor is not None:
        reduced = _round_dithered(work / divisor, dither, k_tile or work.shape[-1])
    else:
        # truncate mode: no rounding step exists, so no dither is applied.
        reduced = work

    clipped_mask = (reduced < lo.to(torch.float32)) | (reduced > hi.to(torch.float32))
    if count_clip is not None:
        count_clip[0] += int(clipped_mask.sum().item())
        count_clip[1] += int(q.numel())

    reduced = torch.clamp(reduced, lo.to(torch.float32), hi.to(torch.float32))

    if divisor is not None:
        reduced = reduced * divisor
        # The reconstructed value must still live in the int8 datapath.
        reduced = torch.clamp(torch.round(reduced), -128.0, 127.0)

    return reduced.to(q.dtype)


# ============================================================
# CALIBRATION  (multi-domain, same mix as the analysis script)
# ============================================================

DEFAULT_CALIBRATION_DOMAINS = [
    {"name": "general-wiki", "hf_path": "Salesforce/wikitext", "hf_config": "wikitext-2-raw-v1",
     "split": "train", "text_field": "text", "weight": 0.25, "streaming": False},
    {"name": "web-c4", "hf_path": "allenai/c4", "hf_config": "en",
     "split": "train", "text_field": "text", "weight": 0.25, "streaming": True},
    {"name": "code-python", "hf_path": "codeparrot/codeparrot-clean-valid", "hf_config": None,
     "split": "train", "text_field": "content", "weight": 0.20, "streaming": True},
    {"name": "math-gsm8k", "hf_path": "openai/gsm8k", "hf_config": "main",
     "split": "train", "join_fields": ["question", "answer"], "weight": 0.15, "streaming": False},
    {"name": "instruct-dolly", "hf_path": "databricks/databricks-dolly-15k", "hf_config": None,
     "split": "train", "join_fields": ["instruction", "context", "response"],
     "weight": 0.15, "streaming": False},
]

# Cheapest possible calibration: one small local-ish corpus, no
# streaming downloads. Note this makes calibration and evaluation
# same-domain (WikiText train vs test), which is a friendlier setting
# than the multi-domain mix -- fine for iterating on the harness, but
# report the multi-domain numbers as the real result.
WIKITEXT_ONLY_DOMAINS = [
    {"name": "general-wiki", "hf_path": "Salesforce/wikitext", "hf_config": "wikitext-2-raw-v1",
     "split": "train", "text_field": "text", "weight": 1.0, "streaming": False},
]

_DOMAIN_TEXT_CACHE = {}


def _get_domain_full_text(domain, max_examples_streaming=50_000):
    key = domain["name"]
    if key in _DOMAIN_TEXT_CACHE:
        print(f"  Reusing cached text for '{key}'.")
        return _DOMAIN_TEXT_CACHE[key]

    hf_config = domain.get("hf_config")
    streaming = bool(domain.get("streaming", False))
    print(f"  Loading calibration domain '{key}': {domain['hf_path']}"
          f"{(' (' + hf_config + ')') if hf_config else ''}{' [streaming]' if streaming else ''} ...")

    kw = {"split": domain["split"], "streaming": streaming}
    ds = load_dataset(domain["hf_path"], hf_config, **kw) if hf_config else load_dataset(domain["hf_path"], **kw)

    join_fields = domain.get("join_fields")
    text_field = domain.get("text_field")

    def to_text(ex):
        if join_fields:
            return "\n".join(str(ex[f]) for f in join_fields if ex.get(f))
        return ex.get(text_field, "")

    texts = []
    if streaming:
        for i, ex in enumerate(ds, start=1):
            t = to_text(ex)
            if t and t.strip():
                texts.append(t)
            if i % 500 == 0:
                print(f"    ...'{key}': scanned {i}, {len(texts)} usable")
            if len(texts) >= max_examples_streaming:
                break
    else:
        texts = [t for t in (to_text(ex) for ex in ds) if t and t.strip()]

    if not texts:
        raise RuntimeError(f"Domain '{key}' yielded no usable text.")

    full = "\n\n".join(texts)
    _DOMAIN_TEXT_CACHE[key] = full
    return full


def allocate_trials_largest_remainder(weights, num_trials):
    if not weights:
        return []
    exact = [w * num_trials for w in weights]
    floors = [int(math.floor(x)) for x in exact]
    remainders = [x - f for x, f in zip(exact, floors)]
    leftover = num_trials - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: remainders[i], reverse=True)
    alloc = floors[:]
    for i in order[:leftover]:
        alloc[i] += 1
    assert sum(alloc) == num_trials
    return alloc


def get_multi_domain_calibration(tokenizer, domains, num_trials=40, seq_len=512, seed=1234):
    weights = [d["weight"] for d in domains]
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("Calibration weights must sum to a positive number.")
    if abs(total_w - 1.0) > 1e-6:
        print(f"  NOTE: weights sum to {total_w:.3f}, normalizing.")
        weights = [w / total_w for w in weights]

    per_domain = allocate_trials_largest_remainder(weights, num_trials)
    print(f"Building calibration set ({num_trials} trials, seq_len={seq_len}):")
    for d, n in zip(domains, per_domain):
        print(f"    {d['name']:<16s}: {n:3d} trials (weight={d['weight']:.2f})")

    samples, provenance = [], []
    for d, n_needed in zip(domains, per_domain):
        if n_needed <= 0:
            continue
        ids = tokenizer(_get_domain_full_text(d), return_tensors="pt").input_ids[0]
        n_possible = ids.numel() // seq_len
        if n_possible < n_needed:
            raise ValueError(
                f"Domain '{d['name']}' needs {n_needed} blocks of {seq_len} tokens "
                f"but only has {n_possible}."
            )
        rng = random.Random(f"{seed}-{d['name']}")
        blocks = list(range(n_possible))
        rng.shuffle(blocks)
        for b in blocks[:n_needed]:
            samples.append(ids[b * seq_len:(b + 1) * seq_len].unsqueeze(0))
            provenance.append(d["name"])

    pool_rng = random.Random(f"{seed}-pool-order")
    combined = list(zip(samples, provenance))
    pool_rng.shuffle(combined)
    samples, provenance = (list(x) for x in zip(*combined))

    print(f"Calibration set assembled: {len(samples)} trials.")
    for name, c in Counter(provenance).items():
        print(f"    {name}: {c}")
    return samples


# ============================================================
# WIKITEXT-2 PERPLEXITY
# ============================================================

_WIKITEXT_TEST_CACHE = {}


def wikitext2_token_ids(tokenizer):
    key = id(tokenizer)
    if key in _WIKITEXT_TEST_CACHE:
        return _WIKITEXT_TEST_CACHE[key]
    print("  Loading WikiText-2 test split ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    _WIKITEXT_TEST_CACHE[key] = ids
    print(f"  WikiText-2 test: {ids.numel()} tokens.")
    return ids


@torch.no_grad()
def eval_wikitext_ppl(model, ids, seq_len, device, max_windows=None, label=""):
    """
    Standard non-overlapping-window perplexity. Each window contributes
    (seq_len - 1) predicted tokens; NLL is token-weighted so the result
    is a true corpus perplexity rather than a mean of per-window means.
    """
    n_windows = ids.numel() // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    total_nll, total_tokens = 0.0, 0
    for i in range(n_windows):
        chunk = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
        out = model(input_ids=chunk, labels=chunk, use_cache=False)
        total_nll += float(out.loss.float().item()) * (seq_len - 1)
        total_tokens += seq_len - 1
        if (i + 1) % 20 == 0:
            running = math.exp(total_nll / total_tokens)
            print(f"    [{label}] window {i + 1}/{n_windows}  running PPL={running:.4f}")

    if total_tokens == 0:
        raise RuntimeError("No evaluation windows -- seq_len larger than the corpus?")
    return math.exp(total_nll / total_tokens)


# ============================================================
# DEPTH-WISE ERROR LOCALIZATION
#
# Per-tensor clip rate and SNR say WHICH tensor is damaged. They do
# not say whether that damage matters, because error injected at
# layer 3 either dies out or compounds through the remaining 27
# layers. This measures the propagation directly: cache the
# baseline's hidden states for a couple of eval windows, re-run the
# same windows with the quantizer live, and take the relative L2
# deviation at every layer boundary.
#
# Reading the curve:
#   flat / decaying  -> the model absorbs the perturbation; PPL will
#                       barely move even if some tensors look ugly
#   steadily rising  -> error compounds with depth; the culprit is
#                       wherever the curve first departs from zero
#   sharp step       -> one layer is responsible; cross-reference its
#                       per-tensor SNR to find which projection
#
# Logit KL is reported alongside because PPL is an average and hides
# localized blowups: a scheme can leave PPL almost unchanged while
# scrambling the distribution on a small subset of tokens. KL over
# the same windows catches that, and per-position NLL delta names
# the specific worst tokens.
#
# MEMORY: baseline hidden states and log-probs are held on CPU for
# --diag-windows windows only (2 by default). Log-probs dominate --
# roughly seq_len * vocab * 2 bytes per window -- so raise the window
# count carefully on large-vocab models.
# ============================================================

@torch.no_grad()
def collect_reference_states(model, ids, seq_len, device, n_windows):
    ref = {"hidden": [], "logprobs": [], "nll": []}
    for i in range(n_windows):
        chunk = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
        out = model(input_ids=chunk, use_cache=False, output_hidden_states=True)
        ref["hidden"].append([h[0].detach().float().cpu() for h in out.hidden_states])
        lp = torch.log_softmax(out.logits[0].detach().float(), dim=-1)
        ref["logprobs"].append(lp.to(torch.float16).cpu())
        tgt = chunk[0, 1:]
        ref["nll"].append((-lp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)).cpu())
        del out, lp
    return ref


@torch.no_grad()
def compare_against_reference(model, ids, seq_len, device, ref):
    """
    Returns (per_layer_relative_l2, mean_kl_nats, worst_positions).
    per_layer_relative_l2[l] = ||h_q[l] - h_b[l]|| / ||h_b[l]||
    """
    n_layers = len(ref["hidden"][0])
    num = [0.0] * n_layers
    den = [0.0] * n_layers
    kl_sum, kl_n = 0.0, 0
    nll_delta = []

    for i in range(len(ref["hidden"])):
        chunk = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
        out = model(input_ids=chunk, use_cache=False, output_hidden_states=True)

        for l, h in enumerate(out.hidden_states):
            hq = h[0].detach().float().cpu()
            hb = ref["hidden"][i][l]
            num[l] += float(((hq - hb) ** 2).sum().item())
            den[l] += float((hb ** 2).sum().item())

        lq = torch.log_softmax(out.logits[0].detach().float(), dim=-1)
        lb = ref["logprobs"][i].float()
        # KL(baseline || quantized), the direction that penalizes the
        # quantized model for putting low mass where the reference is
        # confident.
        pb = lb.exp()
        kl_sum += float((pb * (lb - lq.cpu())).sum(dim=-1).sum().item())
        kl_n += lb.shape[0]

        tgt = chunk[0, 1:]
        nll_q = (-lq[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)).cpu()
        d = nll_q - ref["nll"][i]
        for pos in range(d.numel()):
            nll_delta.append((float(d[pos].item()), i, pos))

        del out, lq, lb, pb

    rel = [math.sqrt(num[l] / den[l]) if den[l] > 0 else 0.0 for l in range(n_layers)]
    nll_delta.sort(reverse=True)
    return rel, (kl_sum / kl_n if kl_n else float("nan")), nll_delta[:10]


def print_depth_report(rel, kl, worst):
    print("  Depth-wise relative L2 deviation of hidden states:")
    peak = max(rel) if rel else 0.0
    for l, r in enumerate(rel):
        bar = "#" * int(round(40 * (r / peak))) if peak > 0 else ""
        print(f"    layer {l:3d}: {r:10.3e}  {bar}")
    if peak > 0:
        first = next((l for l, r in enumerate(rel) if r > 0.01 * peak), None)
        print(f"  Deviation first becomes non-negligible at layer {first}; "
              f"peak {peak:.3e} at layer {rel.index(peak)}.")
        growth = rel[-1] / rel[first] if first is not None and rel[first] > 0 else float("nan")
        print(f"  Growth from first-affected layer to output: {growth:.2f}x "
              f"({'compounding' if growth > 2 else 'absorbed'}).")
    print(f"  Mean logit KL(baseline || quantized) = {kl:.6f} nats")
    print("  Worst per-token NLL regressions (window, position, delta):")
    for d, w, pos in worst[:5]:
        print(f"    window {w}, pos {pos:4d}: {d:+.4f} nats")


# ============================================================
# UNQUANTIZED REFERENCE
#
# The W8A8 checkpoint is the baseline every arm is measured
# against -- that comparison is what isolates THIS scheme's cost.
# But W8A8 is itself already a lossy step, so a delta of +0.15 PPL
# means something different depending on whether the W8A8 -> fp16
# gap is 0.02 or 0.8. This measures that gap so the tiling deltas
# can be read on the right scale.
#
# Deliberately uses the QUANTIZED model's tokenizer so the
# evaluation windows are token-identical between the two models;
# otherwise the perplexities are not comparable at all.
# ============================================================

@torch.no_grad()
def eval_fp16_reference(ref_model_name, tokenizer, eval_seq_len, max_windows):
    print("\n" + "=" * 90)
    print(f"UNQUANTIZED REFERENCE: {ref_model_name}")
    print("=" * 90)

    ids = wikitext2_token_ids(tokenizer)
    model = load_causal_lm(ref_model_name, torch.float16).eval()
    try:
        ref_vocab = model.get_input_embeddings().weight.shape[0]
        if ids.max().item() >= ref_vocab:
            raise RuntimeError(
                f"Token ids exceed the reference model's vocab ({ref_vocab}); the two "
                f"models do not share a tokenizer, so their perplexities are not "
                f"comparable. Pass a --fp16-ref that matches the quantized checkpoint."
            )
        device = next(model.parameters()).device
        ppl = eval_wikitext_ppl(model, ids, eval_seq_len, device, max_windows, label="fp16-ref")
        print(f"\n  Unquantized (fp16) WikiText-2 PPL = {ppl:.4f}")
        return ppl
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ============================================================
# MAIN HARNESS
# ============================================================

ARMS = ["baseline", "force8", "w_only", "a_only", "both"]


class WxAErrorEvaluator:

    def __init__(self, model_name, k_sizes, calib_domains, pair_mode="bucket",
                 sim_mode="shift", bit_stat="mean",
                 dither="none", dither_side="weights", bit_percentile=99.0):
        self.model_name = model_name
        self.k_sizes = k_sizes
        self.calib_domains = calib_domains
        self.pair_mode = pair_mode
        self.sim_mode = sim_mode
        self.bit_stat = bit_stat
        self.dither = dither
        self.dither_side = dither_side
        self.bit_percentile = bit_percentile
        self.neg128_counts = {"weights": 0, "activations": 0}
        self._patch_was_hit = False
        self.fp16_ref_ppl = None
        self.weight_err = {}

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = load_causal_lm(model_name, "auto").eval()

        meta = [n for n, p in self.model.named_parameters() if p.device.type == "meta"]
        if meta:
            raise RuntimeError(
                f"{len(meta)} parameter(s) still on 'meta' after loading (e.g. '{meta[0]}')."
            )

        self.device = next(self.model.parameters()).device

        self.quant_modules = [
            (n, m) for n, m in self.model.named_modules()
            if getattr(m, "quantization_scheme", None) is not None
        ]
        if not self.quant_modules:
            raise RuntimeError(
                "No quantized modules found -- is this a compressed-tensors checkpoint?"
            )

        # ------------------------------------------------------------
        # LAZY DECOMPRESSION -- the reason this warm-up exists
        #
        # The checkpoint loads with quantization_status="compressed":
        # module.weight is int8 at load time, but the FIRST forward pass
        # triggers "Decompressing model", which replaces weight with a
        # bfloat16 tensor equal to (int8 * weight_scale). Snapshotting
        # before that happens captures a representation the model no
        # longer uses, and restoring it later writes integer magnitudes
        # into a float tensor -- a ~1e6 perplexity, not a subtle bug.
        #
        # So force decompression here, then snapshot. The original
        # integers come back exactly as round(w / weight_scale), and
        # weight_scale is per output channel, hence constant within
        # every K-tile -- it does not disturb the LZ statistics.
        # ------------------------------------------------------------
        print("Warm-up forward to trigger lazy decompression before snapshotting ...")
        with torch.no_grad():
            self.model(
                input_ids=torch.tensor([[1, 2, 3, 4, 5]], device=self.device),
                use_cache=False,
            )

        self.native_w8 = {}       # name -> recovered int8 integers (CPU)
        self.weight_scale = {}    # name -> per-output-channel scale (CPU)
        self.weight_dtype = {}    # name -> live weight dtype
        self.native_float = {}    # name -> bf16 original, only where the
                                  #         int8*scale round trip is inexact
        skipped = []
        recon_mismatch = 0

        for name, module in self.quant_modules:
            w = getattr(module, "weight", None)
            scale = getattr(module, "weight_scale", None)

            if not isinstance(w, torch.Tensor) or w.ndim != 2:
                skipped.append(f"{name} (weight not a 2D tensor)")
                continue

            if w.dtype == torch.int8:
                # Never decompressed (some formats stay integer).
                q = w.detach().to(torch.int8).cpu()
                sc = None
            else:
                if not isinstance(scale, torch.Tensor) or scale.numel() == 0:
                    skipped.append(f"{name} (float weight with no weight_scale)")
                    continue
                sc = scale.detach().float().cpu().reshape(-1, 1)
                if sc.shape[0] != w.shape[0]:
                    skipped.append(
                        f"{name} (weight_scale rows {sc.shape[0]} != out_features {w.shape[0]})"
                    )
                    continue
                wf = w.detach().float().cpu()
                q = torch.clamp(torch.round(wf / sc), -128, 127).to(torch.int8)

                # The round trip must reproduce the live weight bit for
                # bit, otherwise every arm silently inherits a constant
                # offset and force8 stops being an identity.
                recon = (q.float() * sc).to(w.dtype)
                if not torch.equal(recon, w.detach().cpu()):
                    recon_mismatch += 1
                    self.native_float[name] = w.detach().cpu().clone()

            self.native_w8[name] = q
            self.weight_scale[name] = sc
            self.weight_dtype[name] = w.dtype
            module._lz_layer_name = name

        if skipped:
            print(f"Skipped {len(skipped)} module(s):")
            for x in skipped[:5]:
                print(f"   - {x}")

        if not self.native_w8:
            raise RuntimeError(
                "No quantized weights could be recovered as integers. Neither an int8 "
                "weight nor a (float weight + weight_scale) pair was found on any "
                "quantized module."
            )

        n_float = sum(1 for n in self.native_w8 if self.weight_dtype[n] != torch.int8)
        print(f"Recovered integer weights for {len(self.native_w8)} tensor(s) "
              f"({n_float} via w/weight_scale after decompression).")
        if recon_mismatch:
            print(f"  NOTE: int8*scale did not reproduce the stored weight exactly for "
                  f"{recon_mismatch} tensor(s); the original float is kept verbatim for "
                  f"those so restore stays exact.")

        self.quant_modules = [(n, m) for n, m in self.quant_modules if n in self.native_w8]
        self.module_by_name = dict(self.quant_modules)
        self.name_by_id = {id(m): n for n, m in self.quant_modules}

        print(f"Quantized modules with native int8 2D weights: {len(self.native_w8)}")

        # frozen_bits[k_tile][name] = {"w": [...], "a": [...], "bucket": [...]}
        self.frozen_bits = {}
        # shape_info[name] = (M, K, N_calib_rows)
        self.shape_info = {}

    # ---------- weight state management ----------

    def _reconstruct(self, name):
        """
        The live weight tensor corresponding to this tensor's stored
        integers: int8 * per-channel scale, cast to the dtype the model
        actually uses. Where the round trip was found inexact at load,
        the verbatim original is returned instead.
        """
        if name in self.native_float:
            return self.native_float[name]
        q = self.native_w8[name]
        sc = self.weight_scale[name]
        if sc is None:
            return q
        return (q.float() * sc).to(self.weight_dtype[name])

    def _write_back(self, name, module, q_int):
        """Write quantized integers back through the scale into the live weight."""
        sc = self.weight_scale[name]
        if sc is None:
            out = q_int.to(torch.int8)
        else:
            out = (q_int.float() * sc).to(self.weight_dtype[name])
        module.weight.data.copy_(out.to(module.weight.device))

    def restore_native_weights(self):
        for name, module in self.quant_modules:
            module.weight.data.copy_(
                self._reconstruct(name).to(module.weight.device)
            )

    def verify_restore_is_identity(self):
        """
        Restoring must be a no-op on an untouched model. If it is not,
        every arm inherits a constant perturbation and the baseline
        itself is wrong -- which is exactly the failure that produced a
        seven-figure perplexity before the snapshot point was fixed.
        """
        dev = self.device
        ids = torch.tensor([[1, 2, 3, 4, 5]], device=dev)
        with torch.no_grad():
            before = self.model(input_ids=ids, use_cache=False).logits.clone()
            self.restore_native_weights()
            after = self.model(input_ids=ids, use_cache=False).logits
            ok = torch.equal(before, after)
            max_d = float((before - after).abs().max().item())
        print(f"Restore-is-identity check: {'PASS' if ok else 'FAIL'} "
              f"(max logit delta {max_d:.3e})")
        if not ok:
            raise RuntimeError(
                "restore_native_weights() changes the model's output on an untouched "
                "model. The stored integer/scale pair does not reproduce the live "
                "weights, so the baseline and every arm would be measured against a "
                "corrupted model. Refusing to continue."
            )
        return ok

    def apply_weight_bounds(self, params_by_name):
        """
        Quantize each weight tensor with its frozen per-tile quantizer.
        Operates on the recovered INTEGERS, then writes back through the
        per-channel scale, so the tile quantizer sees the same integer
        domain the leading-zero statistics were derived from.
        Returns (n_clipped, n_total) across the model.
        """
        stats = [0, 0]
        self.weight_err = {}
        for name, module in self.quant_modules:
            w = self.native_w8[name]
            params = params_by_name.get(name)
            if params is None:
                module.weight.data.copy_(
                    self._reconstruct(name).to(module.weight.device)
                )
                continue

            per = [0, 0]
            out = apply_tile_quant(
                w.to(torch.int16), params, count_clip=per,
                dither=DITHER_CFG["weights"], k_tile=DITHER_CFG["k_tile"],
            )
            stats[0] += per[0]
            stats[1] += per[1]

            d = out.to(torch.float32) - w.to(torch.float32)
            sq_err = float((d ** 2).sum().item())
            sq_sig = float((w.to(torch.float32) ** 2).sum().item())
            self.weight_err[name] = {
                "clip_rate": per[0] / per[1] if per[1] else 0.0,
                "rmse": math.sqrt(sq_err / max(d.numel(), 1)),
                "max_err": float(d.abs().max().item()),
                "snr_db": (10.0 * math.log10(sq_sig / sq_err)) if sq_err > 0 else float("inf"),
            }
            self._write_back(name, module, out)
        return stats[0], stats[1]

    # ---------- calibration / freezing ----------

    def calibrate(self, num_trials, seq_len):
        samples = get_multi_domain_calibration(
            self.tokenizer, self.calib_domains, num_trials=num_trials, seq_len=seq_len
        )

        print("\n=== Accumulating native A8 leading-zero statistics ===")
        print("  (streaming per-column sums -- no activation tensors retained)")
        ACT_LZ_COLSUM.clear()
        ACT_ROWS.clear()
        ACT_ABSMAX.clear()
        ACT_VALSUM.clear()
        ACT_LZ_HIST.clear()
        ACT_NEG128.clear()
        MODE["capture"], MODE["apply"] = True, False
        try:
            with patched_quant_path():
                for i, ids in enumerate(samples):
                    if (i + 1) % 5 == 0 or i == 0:
                        print(f"  trial {i + 1}/{len(samples)}")
                    with torch.no_grad():
                        self.model(input_ids=ids.to(self.device), use_cache=False)
                    del ids
        finally:
            MODE["capture"] = False

        print(f"Accumulated statistics for {len(ACT_LZ_COLSUM)} modules.")
        total_neg128_act = sum(ACT_NEG128.values())
        print(f"  Exact -128 activation values seen during calibration: {total_neg128_act}")

        print("Computing weight LZ column sums and magnitudes ...")
        weight_stats = {}
        w_neg128 = 0
        for name, w in self.native_w8.items():
            w16 = w.to(torch.int16)
            weight_stats[name] = (
                lz_colsum(w16)[0],
                w16.shape[0],
                w16.abs().amax(dim=0).to(torch.int16),
                lz_hist(w16),
            )
            w_neg128 += int((w16 == -128).sum().item())
        print(f"  Exact -128 weight values in this checkpoint: {w_neg128}")
        self.neg128_counts = {"weights": w_neg128, "activations": total_neg128_act}

        # Channel DC levels decide whether dithering can pay off at all.
        dc_ratios = []
        for mid, vsum in ACT_VALSUM.items():
            mean_ch = (vsum / max(ACT_ROWS[mid], 1)).abs()
            scale = ACT_ABSMAX[mid].to(torch.float64).clamp(min=1.0)
            dc_ratios.append(float((mean_ch / scale).mean().item()))
        if dc_ratios:
            dc = sum(dc_ratios) / len(dc_ratios)
            print(f"\n  Mean activation DC ratio |x_bar| / max|x| = {dc:.4f}")
            print("  (Dithering acts only on the output-error term Sum(w'-w)*x_bar.")
            if dc < 0.02:
                print("   This is near zero, so error-feedback dithering will likely HURT:")
                print("   it doubles per-element MSE with no bias term to cancel.)")
            else:
                print("   This is substantial, so error-feedback dithering has something")
                print("   to cancel and is worth testing against --dither none.)")

        # The histogram supersedes the running LZ sums, so verify once
        # that it reproduces the mean-based bits exactly before relying
        # on it for the percentile statistic too.
        _chk = next(iter(self.native_w8))
        _cs, _rows, _am, _h = weight_stats[_chk]
        _a = bits_from_lz_colsum(_cs, _rows, 128)
        _b = bits_from_lz_hist(_h, 128, "mean")
        print(f"\n  Histogram/running-sum agreement on mean bits ({_chk}): "
              f"{'exact' if _a == _b else 'MISMATCH'}")
        if _a != _b:
            print(f"    running-sum: {_a[:8]}")
            print(f"    histogram:   {_b[:8]}")
            raise RuntimeError(
                "LZ histogram disagrees with the running-sum mean. One of the two "
                "accumulators is wrong; refusing to proceed."
            )

        print("\n=== Deriving and FREEZING per-tile bit-widths ===")
        _stat_desc = (f"percentile p{self.bit_percentile:g}"
                      if self.bit_stat == "percentile" else self.bit_stat)
        print(f"  bit statistic = {_stat_desc}   simulation mode = {self.sim_mode}")
        print("  (derived on calibration data only -- never on WikiText)")
        if self.bit_stat == "mean" and self.sim_mode == "truncate":
            print("\n  !!! mean-LZ bits applied with shift=0. Bits chosen from the MEAN")
            print("  !!! leading-zero count do not cover the tile's largest elements, and")
            print("  !!! truncate mode has no shift to compensate, so the entire high end")
            print("  !!! of every tile saturates. Expect this arm to look far worse than")
            print("  !!! any real b-bit hardware mode. Use --sim-mode shift for the")
            print("  !!! faithful comparison, or --bit-stat min for the lossless bound.\n")

        for k_tile in self.k_sizes:
            per_tensor = {}
            skipped = []
            for name, w_int8 in self.native_w8.items():
                M, K = w_int8.shape
                mid = id(self.module_by_name[name])

                a_colsum = ACT_LZ_COLSUM.get(mid)
                if a_colsum is None:
                    skipped.append(f"{name} (no activations captured)")
                    continue
                if K % k_tile != 0:
                    skipped.append(f"{name} (K={K} not divisible by {k_tile})")
                    continue
                if a_colsum.numel() != K:
                    skipped.append(f"{name} (activation K={a_colsum.numel()} != weight K={K})")
                    continue

                w_colsum, w_rows, w_absmax, w_hist = weight_stats[name]
                a_absmax = ACT_ABSMAX[mid].cpu()
                a_hist = ACT_LZ_HIST[mid].cpu()

                w_bits = bits_from_lz_hist(
                    w_hist, k_tile, self.bit_stat, self.bit_percentile)
                a_bits = bits_from_lz_hist(
                    a_hist, k_tile, self.bit_stat, self.bit_percentile)

                bucket = [map_to_bucket_bits(w, a) for w, a in zip(w_bits, a_bits)]

                per_tensor[name] = {
                    "w": w_bits, "a": a_bits, "bucket": bucket,
                    "w_absmax": w_absmax, "a_absmax": a_absmax,
                }
                self.shape_info[name] = (M, K, ACT_ROWS[mid])

            # How the promotion actually landed. If one bucket holds
            # nearly everything, that bucket's shift depth is what the
            # PPL will be about.
            if per_tensor:
                dist = Counter()
                raw = Counter()
                for b in per_tensor.values():
                    dist.update(b["bucket"])
                    raw.update(max(w, a) for w, a in zip(b["w"], b["a"]))
                tot = sum(dist.values())
                pieces = ", ".join(
                    f"b={k}: {v / tot * 100:.1f}%" for k, v in sorted(dist.items())
                )
                print(f"      buckets -> {pieces}")
                promoted = sum(v for k, v in raw.items() if k not in SUPPORTED_BUCKET_BITS)
                if promoted:
                    print(f"      ({promoted / tot * 100:.1f}% of tiles promoted "
                          f"to a wider supported mode)")

            self.frozen_bits[k_tile] = per_tensor
            print(f"  k_tile={k_tile:<4d}: froze bit-vectors for {len(per_tensor)} tensor(s)"
                  + (f", skipped {len(skipped)}" if skipped else ""))
            for s in skipped[:5]:
                print(f"      - skipped {s}")
            if len(skipped) > 5:
                print(f"      - ... and {len(skipped) - 5} more")

        del weight_stats
        ACT_LZ_COLSUM.clear()
        ACT_ROWS.clear()
        ACT_ABSMAX.clear()
        ACT_VALSUM.clear()
        ACT_LZ_HIST.clear()
        ACT_NEG128.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_weighted_effective_bits(self, k_tile):
        """
        MAC-weighted mean bucket bit-width, plus the fraction of model
        weight parameters the tiled scheme actually covers.

        Tensors skipped during calibration (K not divisible by k_tile,
        or no captured activations) stay at native int8 during run() but
        are excluded from this average. So the returned effective-bits
        figure is "effective bits AMONG TILED LAYERS", not model-wide --
        read the Pareto x-axis together with the coverage number, which
        is also written to the CSV.
        """
        num = den = 0.0
        for name, bits in self.frozen_bits[k_tile].items():
            M, K, N = self.shape_info[name]
            macs_per_tile = float(N) * float(k_tile) * float(M)
            for b in bits["bucket"]:
                num += macs_per_tile * b
                den += macs_per_tile

        # MAC-weighted mean shift depth. With mean-LZ bits this, not the
        # nominal bit-width, is what varies across tile sizes: b stays
        # pinned near the mean while m grows with tile size, so s = m - b
        # deepens and the rounding coarsens. Nominal bits are nearly
        # constant across k_tile, so plotting against them is misleading.
        snum = sden = 0.0
        for name, bits in self.frozen_bits[k_tile].items():
            M, K, N = self.shape_info[name]
            macs_per_tile = float(N) * float(k_tile) * float(M)
            am = bits["a_absmax"]
            tmax = am.reshape(K // k_tile, k_tile).amax(dim=1)
            m = magnitude_bits(tmax) + 1
            for b, mm in zip(bits["bucket"], m.tolist()):
                snum += macs_per_tile * max(0, int(mm) - int(b))
                sden += macs_per_tile
        self._mean_shift = (snum / sden) if sden else float("nan")

        covered = sum(self.native_w8[n].numel() for n in self.frozen_bits[k_tile])
        total = sum(w.numel() for w in self.native_w8.values())
        coverage = covered / total if total else 0.0
        return ((num / den) if den else float("nan")), coverage

    # ---------- bound construction per arm ----------

    def _bit_vectors(self, k_tile, which):
        """which: 'bucket' | 'w' | 'a'  -- selects which frozen vector to use."""
        return {name: bits[which] for name, bits in self.frozen_bits[k_tile].items()}

    def build_params(self, k_tile, arm):
        """
        Returns (weight_params_by_name, act_params_by_module_id).
        Either may be empty, meaning "leave that side at native int8".

        Note on w_only / a_only in the default bucket pair-mode: the side
        being touched receives the MATCHED-BUCKET vector, not its own
        error-optimal bit vector. These arms isolate each side's
        contribution to the SAME headline scheme; they are not "weights
        at their own optimal precision". Use --pair-mode exact if you
        want each side driven by its own derived bits.
        """
        if arm == "baseline":
            return {}, {}

        frozen = self.frozen_bits[k_tile]

        if arm == "force8":
            w_src = a_src = {n: [8] * len(b["bucket"]) for n, b in frozen.items()}
        elif self.pair_mode == "exact":
            w_src = {n: b["w"] for n, b in frozen.items()}
            a_src = {n: b["a"] for n, b in frozen.items()}
        else:
            w_src = a_src = {n: b["bucket"] for n, b in frozen.items()}

        w_params, a_params = {}, {}
        if arm in ("force8", "w_only", "both"):
            for n, v in w_src.items():
                q = build_tile_quant(v, k_tile, frozen[n]["w_absmax"], self.sim_mode)
                if q is not None:
                    w_params[n] = q
        if arm in ("force8", "a_only", "both"):
            for n, v in a_src.items():
                q = build_tile_quant(v, k_tile, frozen[n]["a_absmax"], self.sim_mode)
                if q is not None:
                    a_params[id(self.module_by_name[n])] = q
        return w_params, a_params

    # ---------- the run ----------

    def run(self, eval_seq_len, max_windows, arms, out_rows,
            diagnose=False, diag_windows=2, tensor_csv=None):
        ids = wikitext2_token_ids(self.tokenizer)

        print("\n" + "=" * 90)
        print("BASELINE (native W8A8, untouched)")
        print("=" * 90)
        self.verify_restore_is_identity()
        ACT_BOUNDS.clear()
        MODE["capture"], MODE["apply"] = False, False
        baseline_ppl = eval_wikitext_ppl(
            self.model, ids, eval_seq_len, self.device, max_windows, label="baseline"
        )
        print(f"\n  Baseline (native W8A8) WikiText-2 PPL = {baseline_ppl:.4f}")
        ref = getattr(self, "fp16_ref_ppl", None)
        if ref is not None:
            print(f"  Unquantized fp16 reference          = {ref:.4f}")
            print(f"  Cost of the W8A8 step itself        = {baseline_ppl - ref:+.4f} "
                  f"({(baseline_ppl - ref) / ref * 100:+.2f}%)")
            print("  Every arm delta below is measured against the W8A8 baseline, so")
            print("  compare it against this W8A8-step cost to judge its magnitude.")

        out_rows.append({
            "model": self.model_name, "k_tile": "-", "arm": "baseline",
            "ppl": f"{baseline_ppl:.6f}", "delta_ppl": "0.000000",
            "pct_delta": "0.00", "eff_bits": "8.000",
            "w_clip_rate": "0.000000", "a_clip_rate": "0.000000",
            "coverage": "1.000", "mean_shift": "0.0000",
            "sim_mode": self.sim_mode, "bit_stat": self.bit_stat,
            "dither": "none/none",
            "buckets": "-".join(str(b) for b in SUPPORTED_BUCKET_BITS),
        })

        ref_states = None
        if diagnose:
            print(f"\n  Caching baseline hidden states / log-probs for "
                  f"{diag_windows} window(s) (error localization) ...")
            ref_states = collect_reference_states(
                self.model, ids, eval_seq_len, self.device, diag_windows
            )

        for k_tile in self.k_sizes:
            if not self.frozen_bits.get(k_tile):
                print(f"\n!!! No frozen bit-vectors for k_tile={k_tile}; skipping.")
                continue

            eff_bits, coverage = self.compute_weighted_effective_bits(k_tile)
            print("\n" + "=" * 90)
            print(f"K-TILE = {k_tile}   (effective bits among tiled layers = {eff_bits:.3f}, "
                  f"mean shift depth = {getattr(self, '_mean_shift', float('nan')):.3f} bits, "
                  f"weight coverage = {coverage * 100:.1f}%)")
            print("=" * 90)

            for arm in arms:
                if arm == "baseline":
                    continue

                print(f"\n--- arm: {arm} ---")
                DITHER_CFG["k_tile"] = k_tile
                if arm == "force8" or self.sim_mode == "truncate":
                    # force8 must stay an exact identity, and truncate has
                    # no rounding step for a dither to act on.
                    DITHER_CFG["weights"] = DITHER_CFG["activations"] = "none"
                else:
                    DITHER_CFG["weights"] = (
                        self.dither if self.dither_side in ("weights", "both") else "none")
                    DITHER_CFG["activations"] = (
                        self.dither if self.dither_side in ("activations", "both") else "none")

                w_bounds, a_bounds = self.build_params(k_tile, arm)

                self.restore_native_weights()
                w_clipped, w_total = (0, 0)
                if w_bounds:
                    w_clipped, w_total = self.apply_weight_bounds(w_bounds)

                ACT_BOUNDS.clear()
                ACT_CLIP_STATS.clear()
                ACT_ERR.clear()
                ACT_BOUNDS.update(a_bounds)

                MODE["capture"] = False
                MODE["apply"] = bool(a_bounds)
                PATCH_HITS["forward_quantize"] = 0
                PATCH_HITS["round"] = 0

                try:
                    with patched_quant_path():
                        ppl = eval_wikitext_ppl(
                            self.model, ids, eval_seq_len, self.device,
                            max_windows, label=f"k{k_tile}/{arm}",
                        )
                finally:
                    MODE["apply"] = False
                self._patch_was_hit = PATCH_HITS["round"] > 0

                a_clipped = sum(v[0] for v in ACT_CLIP_STATS.values())
                a_total = sum(v[1] for v in ACT_CLIP_STATS.values())

                w_rate = (w_clipped / w_total) if w_total else 0.0
                a_rate = (a_clipped / a_total) if a_total else 0.0
                delta = ppl - baseline_ppl
                pct = delta / baseline_ppl * 100 if baseline_ppl else float("nan")

                print(f"\n  PPL={ppl:.4f}   dPPL={delta:+.4f} ({pct:+.2f}%)"
                      f"   W clip={w_rate * 100:.4f}%   A clip={a_rate * 100:.4f}%")

                if arm == "force8":
                    n128 = self.neg128_counts
                    if abs(delta) > 1e-3:
                        print("\n  !!! FORCE8 SANITY CHECK FAILED !!!")
                        print("  At b=8 the quantizer is built as an exact pass-through (no bound")
                        print("  is manufactured, so -128 survives), which means PPL MUST equal")
                        print("  baseline. A nonzero delta therefore indicates the quantizer is")
                        print("  not landing on the live datapath -- most likely the checkpoint")
                        print("  decompresses to fp16 and caches it at load time, so mutating")
                        print("  module.weight does nothing. Every number below is untrustworthy")
                        print("  until this is fixed.")
                        print(f"  (-128 values seen: {n128['weights']} in weights, "
                              f"{n128['activations']} in calibration activations -- these are")
                        print("   handled by construction and are NOT the cause.)")
                    else:
                        print("  force8 sanity check PASSED (quantizer is on the live datapath).")
                    if not self._patch_was_hit:
                        print("  !!! WARNING: the compressed_tensors patch was never invoked during")
                        print("  !!! this arm. The library internals may have been renamed by a")
                        print("  !!! version bump; activation-side results are meaningless.")

                # Top offenders by activation clip rate -- usually explains
                # the whole PPL delta on its own.
                if a_total:
                    worst = sorted(
                        ((v[0] / v[1] if v[1] else 0.0, self.name_by_id.get(mid, str(mid)))
                         for mid, v in ACT_CLIP_STATS.items()),
                        reverse=True,
                    )[:5]
                    print("  Highest activation clip rates:")
                    for rate, nm in worst:
                        print(f"    {rate * 100:7.3f}%  {nm}")

                if diagnose and ref_states is not None and arm != "force8":
                    print("\n  --- error localization ---")
                    rel, kl, worst = compare_against_reference(
                        self.model, ids, eval_seq_len, self.device, ref_states
                    )
                    print_depth_report(rel, kl, worst)
                    self._report_tensor_errors(k_tile, arm, tensor_csv)

                out_rows.append({
                    "model": self.model_name, "k_tile": k_tile, "arm": arm,
                    "ppl": f"{ppl:.6f}", "delta_ppl": f"{delta:+.6f}",
                    "pct_delta": f"{pct:+.4f}",
                    "eff_bits": ("8.000" if arm == "force8" else f"{eff_bits:.3f}"),
                    "w_clip_rate": f"{w_rate:.6f}", "a_clip_rate": f"{a_rate:.6f}",
                    "coverage": f"{coverage:.3f}",
                    "mean_shift": f"{getattr(self, '_mean_shift', float('nan')):.4f}",
                    "sim_mode": self.sim_mode, "bit_stat": self.bit_stat,
                    "dither": DITHER_CFG["weights"] + "/" + DITHER_CFG["activations"],
                    "buckets": "-".join(str(b) for b in SUPPORTED_BUCKET_BITS),
                })

        self.restore_native_weights()
        ACT_BOUNDS.clear()

    def _report_tensor_errors(self, k_tile, arm, tensor_csv=None):
        """
        Per-tensor error table, sorted by activation SNR (worst first).
        SNR localizes damage better than clip rate: a tensor can clip
        rarely but catastrophically, or often but harmlessly.
        """
        rows = []
        for name, module in self.quant_modules:
            mid = id(module)
            e = ACT_ERR.get(mid)
            a_snr = float("inf")
            a_rmse = a_max = 0.0
            if e and e[0] > 0:
                a_snr = 10.0 * math.log10(e[1] / e[0]) if e[1] > 0 else float("-inf")
                a_max = e[2]
            c = ACT_CLIP_STATS.get(mid)
            a_clip = (c[0] / c[1]) if c and c[1] else 0.0

            w = self.weight_err.get(name, {})
            rows.append({
                "k_tile": k_tile, "arm": arm, "tensor": name,
                "a_snr_db": a_snr, "a_clip_rate": a_clip, "a_max_err": a_max,
                "w_snr_db": w.get("snr_db", float("inf")),
                "w_clip_rate": w.get("clip_rate", 0.0),
                "w_rmse": w.get("rmse", 0.0),
            })

        finite = [r for r in rows if r["a_snr_db"] != float("inf") or r["w_snr_db"] != float("inf")]
        finite.sort(key=lambda r: min(r["a_snr_db"], r["w_snr_db"]))

        print("\n  Worst tensors by SNR (lower dB = more damage):")
        print(f"    {'tensor':<48s} {'A SNR':>8s} {'A clip':>8s} {'W SNR':>8s} {'W clip':>8s}")
        for r in finite[:12]:
            a = "  inf" if r["a_snr_db"] == float("inf") else f"{r['a_snr_db']:8.2f}"
            wv = "  inf" if r["w_snr_db"] == float("inf") else f"{r['w_snr_db']:8.2f}"
            print(f"    {r['tensor'][:48]:<48s} {a:>8s} "
                  f"{r['a_clip_rate'] * 100:7.3f}% {wv:>8s} {r['w_clip_rate'] * 100:7.3f}%")

        # Aggregate by module type (q_proj / k_proj / mlp.down_proj / ...)
        # so a systematic weakness shows up instead of 200 scattered rows.
        by_kind = defaultdict(list)
        for r in finite:
            kind = r["tensor"].split(".")[-1]
            if r["a_snr_db"] != float("inf"):
                by_kind[kind].append(r["a_snr_db"])
        if by_kind:
            print("\n  Mean activation SNR by projection type:")
            for kind, vals in sorted(by_kind.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
                print(f"    {kind:<20s} {sum(vals) / len(vals):8.2f} dB  ({len(vals)} tensors)")


# ============================================================
# ENTRY POINT
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["RedHatAI/Qwen2.5-7B-quantized.w8a8"])
    p.add_argument("--trials", type=int, default=8, help="Calibration trials, pooled across domains.")
    p.add_argument("--calib-preset", choices=["wikitext", "multi"], default="wikitext",
                   help="'wikitext': WikiText-2 train only -- cheapest, no streaming downloads, "
                        "but same-domain with the eval set. 'multi': the full five-domain mix. "
                        "Overridden by --calib-config if given.")
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--eval-seq-len", type=int, default=512,
                   help="Keep equal to --calib-seq-len unless you intend to study the mismatch: "
                        "activation ranges grow with context, so calibrating at 512 and "
                        "evaluating at 2048 manufactures clipping that the scheme would not "
                        "actually suffer in deployment.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap WikiText-2 eval windows (useful for smoke tests).")
    p.add_argument("--k-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--dither", choices=DITHER_MODES, default="none",
                   help="Rounding dither, active only in shift/rescale sim modes (truncate "
                        "has no rounding step). 'error-feedback' carries the residual along "
                        "K within a tile so the tile's signed error sums to <=0.5 LSB -- it "
                        "loses to plain rounding on zero-mean channels and wins once channel "
                        "DC levels are large. 'stochastic' removes bias but never beats plain "
                        "rounding on RMSE; use it as a control.")
    p.add_argument("--dither-side", choices=["weights", "activations", "both"],
                   default="weights",
                   help="Weights are dithered offline for free. Dithering activations with "
                        "error-feedback runs a k_tile-length sequential loop per module per "
                        "forward and is SLOW; it also needs a PRNG or residual register in "
                        "real hardware.")
    p.add_argument("--dither-seed", type=int, default=1234)
    p.add_argument("--diagnose", action="store_true",
                   help="Localize the error: per-tensor SNR/clip tables, depth-wise "
                        "hidden-state divergence, logit KL, and worst per-token NLL "
                        "regressions. Adds one extra forward pass per arm over "
                        "--diag-windows windows.")
    p.add_argument("--diag-windows", type=int, default=2,
                   help="Windows used for error localization. Baseline log-probs are "
                        "held on CPU at roughly seq_len*vocab*2 bytes each, so raise "
                        "this carefully on large-vocab models.")
    p.add_argument("--fp16-ref", default=None,
                   help="Optional unquantized parent model id (e.g. Qwen/Qwen2.5-7B). "
                        "Measured on the same token windows to establish how much the "
                        "W8A8 step itself already costs, so the tiling deltas can be read "
                        "in context. Loaded and freed before the quantized model, so it "
                        "adds no peak memory. Arm deltas are always computed against the "
                        "W8A8 baseline, never against this.")
    p.add_argument("--sim-mode", choices=SIM_MODES, default="shift",
                   help="What a b-bit mode DOES numerically. 'shift' (default): per-tile "
                        "power-of-two rescale -- the faithful b-bit mode, error is rounding. "
                        "'truncate': keep the low b bits at the native scale, shift=0 -- only "
                        "correct when bits came from --bit-stat min; with mean-LZ bits it "
                        "saturates the entire high end of every tile. 'rescale': per-tile "
                        "affine scale, the most generous mode, needs a real multiplier.")
    p.add_argument("--bucket-bits", type=int, nargs="+", default=None,
                   help="Supported matched-precision modes, e.g. --bucket-bits 2 4 6 8 or "
                        "--bucket-bits 6 8. Must include 8. Tiles are promoted upward to "
                        "the smallest member >= max(w,a), so a coarser set gives better "
                        "perplexity and less compression. Cheapest axis to sweep: no "
                        "recalibration needed, only the mapping changes. Default 2 5 6 8.")
    p.add_argument("--bit-percentile", type=float, default=99.0,
                   help="With --bit-stat percentile: the share of elements each tile's "
                        "bit-width must cover. 100 is equivalent to min (lossless), and "
                        "lower values approach mean. Sweep 99.9 / 99 / 95 / 90 to find "
                        "where perplexity turns.")
    p.add_argument("--bit-stat", choices=["mean", "min", "percentile"], default="mean",
                   help="How per-tile bit-width is chosen. 'mean': mean leading-zero count "
                        "(the aggressive scheme). 'min': minimum LZ, i.e. sized to the tile's "
                        "largest element -- lossless under --sim-mode truncate, so it gives "
                        "the zero-error reference point.")
    p.add_argument("--pair-mode", choices=["bucket", "exact"], default="bucket",
                   help="'bucket': both W and A get max(w,a) mapped to a supported mode "
                        "(the hardware story). 'exact': W gets w_bits, A gets a_bits "
                        "independently -- a strictly harsher lower bound.")
    p.add_argument("--calib-config", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--no-log-file", action="store_true")
    args = p.parse_args()

    if args.calib_config:
        with open(args.calib_config, "r", encoding="utf-8") as f:
            domains = json.load(f)
    elif args.calib_preset == "wikitext":
        domains = WIKITEXT_ONLY_DOMAINS
        print("Calibration: WikiText-2 train only (--calib-preset wikitext).")
        print("  Same-domain with the eval set, so treat the resulting PPL deltas as")
        print("  optimistic. Re-run with --calib-preset multi for the reportable numbers.\n")
    else:
        domains = DEFAULT_CALIBRATION_DOMAINS

    if args.bucket_bits:
        vals = set_bucket_bits(args.bucket_bits)
        print(f"Supported matched-precision modes: {vals}\n")

    if args.dither != "none" and args.sim_mode == "truncate":
        print("!!! --dither has no effect with --sim-mode truncate: there is no rounding")
        print("!!! step to dither, only saturation. Use --sim-mode shift.\n")
    if args.bit_stat == "min" and args.sim_mode == "truncate" and args.dither != "none":
        print("!!! min-LZ + truncate is bit-exact; dithering cannot change it.\n")

    if args.eval_seq_len != args.calib_seq_len:
        print(f"!!! NOTE: calibrating at {args.calib_seq_len} but evaluating at "
              f"{args.eval_seq_len}. Expect inflated activation clipping.\n")

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_stdout = sys.stdout
    log_file = None
    rows = []

    try:
        if not args.no_log_file:
            # One artifact per run: a single timestamped log naming the
            # configuration, so results/ stays readable across many runs.
            tag = f"{args.bit_stat}"
            if args.bit_stat == "percentile":
                tag += f"-p{args.bit_percentile:g}"
            tag += f"_{args.sim_mode}"
            if args.dither != "none":
                tag += f"_dith-{args.dither}"
            log_path = os.path.join(args.results_dir, f"run_{tag}_{ts}.txt")
            log_file = open(log_path, "w", encoding="utf-8")
            sys.stdout = Tee(sys.__stdout__, log_file)
            print(f"Logging to: {log_path}\n")

        for idx, model_name in enumerate(args.models, start=1):
            print("\n" + "#" * 100)
            print(f"##   {idx}/{len(args.models)}  MODEL: {model_name}")
            print("#" * 100 + "\n")

            if args.fp16_ref:
                try:
                    ref_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    ref_ppl = eval_fp16_reference(
                        args.fp16_ref, ref_tok, args.eval_seq_len, args.max_windows
                    )
                    rows.append({
                        "model": model_name, "k_tile": "-", "arm": "fp16_ref",
                        "ppl": f"{ref_ppl:.6f}", "delta_ppl": "", "pct_delta": "",
                        "eff_bits": "16.000", "w_clip_rate": "", "a_clip_rate": "",
                        "coverage": "", "mean_shift": "",
                        "sim_mode": args.sim_mode, "bit_stat": args.bit_stat,
                        "dither": "none/none",
                        "buckets": "-".join(str(b) for b in SUPPORTED_BUCKET_BITS),
                    })
                except Exception as e:
                    print(f"!!! fp16 reference failed ({e}); continuing without it.")
                    traceback.print_exc()

            ev = None
            try:
                ev = WxAErrorEvaluator(model_name, args.k_sizes, domains,
                                       pair_mode=args.pair_mode,
                                       sim_mode=args.sim_mode,
                                       bit_stat=args.bit_stat,
                                       dither=args.dither,
                                       dither_side=args.dither_side,
                                       bit_percentile=args.bit_percentile)
                if args.dither != "none":
                    set_dither_seed(args.dither_seed, ev.device)
                if args.fp16_ref and rows and rows[-1]["arm"] == "fp16_ref":
                    ev.fp16_ref_ppl = float(rows[-1]["ppl"])
                ev.calibrate(args.trials, args.calib_seq_len)
                ev.run(args.eval_seq_len, args.max_windows, args.arms, rows,
                       diagnose=args.diagnose, diag_windows=args.diag_windows)
            except Exception as e:
                print(f"\n!!! ERROR on {model_name}: {e}")
                traceback.print_exc()
            finally:
                if ev is not None:
                    del ev
                ACT_LZ_COLSUM.clear()
                ACT_ROWS.clear()
                ACT_ABSMAX.clear()
                ACT_VALSUM.clear()
                ACT_LZ_HIST.clear()
                ACT_NEG128.clear()
                ACT_BOUNDS.clear()
                ACT_CLIP_STATS.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Final summary table -- the log file is the only artifact, so
        # every number needed to read the result must appear in it.
        if rows:
            print("\n" + "=" * 90)
            print("SUMMARY")
            print("=" * 90)
            hdr = f"{'k_tile':>7s} {'arm':<10s} {'eff bits':>9s} {'PPL':>12s} {'dPPL':>12s} {'%':>10s}"
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                print(f"{str(r['k_tile']):>7s} {r['arm']:<10s} {r['eff_bits']:>9s} "
                      f"{float(r['ppl']):12.4f} {r['delta_ppl'] or '-':>12s} "
                      f"{(r['pct_delta'] + '%') if r['pct_delta'] else '-':>10s}")
            print("-" * len(hdr))
            print(f"config: bit_stat={args.bit_stat}"
                  + (f" p{args.bit_percentile:g}" if args.bit_stat == "percentile" else "")
                  + f", sim_mode={args.sim_mode}"
                  + f", buckets={SUPPORTED_BUCKET_BITS}"
                  + f", dither={args.dither}"
                  + f", calib={args.calib_preset} x{args.trials} @{args.calib_seq_len}"
                  + f", eval @{args.eval_seq_len}")

        print("\n" + "#" * 100)
        print("##   DONE")
        print("#" * 100)

    finally:
        sys.stdout = original_stdout
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()