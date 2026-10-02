#!/usr/bin/env python3
"""
============================================================
ROTATED WxA -> WIKITEXT-2 PERPLEXITY
============================================================

Runs the error harness with a block-diagonal Hadamard applied
along K before quantization, and removed after. Everything else
-- the LZ statistic, the bucket mapping, the tile quantizer, the
eval loop -- comes from the harness unchanged, so any delta
against a non-rotated run is attributable to the rotation alone.

WHY ROTATE
----------
The error in shift mode is the rounding step 2^s, where

    s = m - b

b comes from the LZ statistic (what a typical element needs) and
m from the tile maximum (what the largest element needs). Every
bit of that gap is waste: the bits describe the typical
magnitude while the scale describes the extreme.

A Hadamard spreads each outlier's energy across its block, so the
tile maximum falls toward the tile median and m drops.

WHY IT IS FREE, MATHEMATICALLY
------------------------------
For orthonormal H,

    W x = W (H^T H) x = (W H^T)(H x)

The dot product is unchanged. Quantization noise is also
norm-preserved, so noise measured in the rotated basis equals
noise in the original one.

THE ROUND TRIP, AND WHY IT IS NEEDED
------------------------------------
In deployed hardware the rotation is folded into adjacent layers
and never materializes. Here it must, because the hook returns
integers that compressed_tensors then dequantizes with the scale
it computed itself. Returning values on a different scale would
corrupt the layer. So the simulation is

    rotate -> re-quantize per row -> tile-quantize -> undo the
    rescale -> rotate back -> round to int8

Sylvester Hadamard is symmetric and orthonormal, so H H = I and
the rotation is its own inverse: the round trip is exact and only
the quantization noise survives.

THE PER-ROW RESCALE
-------------------
After rotation the row's maximum is lower than before, so
re-quantizing to int8 in the rotated basis spreads the typical
element further up the range. That rescale is where part of the
benefit is realized -- run --no-rotate to measure how much of it
is the rescale alone and how much is the Hadamard.

EXPECT eff_bits TO RISE. Rotation destroys the near-zero elements
that were producing high LZ and therefore low b. Judge the result
on PPL against eff_bits, not on either alone.

USAGE
-----
  python rotate.py --harness wxax.py --rot-block 16 \\
      --bit-stat mean --sim-mode shift --bucket-bits 2 5 6 8
  python rotate.py --harness wxax.py --no-rotate \\
      --bit-stat mean --sim-mode shift --bucket-bits 2 5 6 8
============================================================
"""

import argparse
import gc
import importlib.util
import math
import os
import sys
import traceback
from datetime import datetime

import torch


# ============================================================
# HARNESS IMPORT
#
# Names checked here are the ones this file actually reaches
# into. A mismatch means the harness has been renamed or an
# older copy is being pointed at, and failing now beats failing
# three hours into a run.
# ============================================================

REQUIRED = (
    "WxAErrorEvaluator", "apply_tile_quant", "lz_hist", "patched_quant_path",
    "set_bucket_bits", "SUPPORTED_BUCKET_BITS", "SIM_MODES", "Tee",
    "MODE", "current_context", "ACT_LZ_HIST", "ACT_ROWS", "ACT_ABSMAX",
    "ACT_LZ_COLSUM", "ACT_VALSUM", "ACT_NEG128",
    "ACT_BOUNDS", "ACT_CLIP_STATS", "orig_round_helpers",
    "_is_a8_dynamic_token", "WIKITEXT_ONLY_DOMAINS",
    "DEFAULT_CALIBRATION_DOMAINS", "compute_elementwise_lz",
)


def load_harness(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"harness not found: {path}")
    spec = importlib.util.spec_from_file_location("_wxa_err", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wxa_err"] = mod
    spec.loader.exec_module(mod)
    missing = [n for n in REQUIRED if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(
            f"'{os.path.basename(path)}' is missing {missing[:4]}"
            + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
            + ". This runner needs the current ERROR harness "
              "(WxAErrorEvaluator / apply_tile_quant), not an older copy."
        )
    return mod


# ============================================================
# HADAMARD
# ============================================================

_H_CACHE = {}


def get_H(n, device, dtype=torch.float32):
    """
    Orthonormal Sylvester Hadamard, cached per (order, device).

    Symmetric and orthonormal, so H @ H == I: one function both
    rotates and un-rotates. Order 1 is the 1x1 identity, which is
    how --no-rotate isolates the per-row rescale.
    """
    key = (n, str(device), dtype)
    if key in _H_CACHE:
        return _H_CACHE[key]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard order must be a power of two; got {n}")
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = (H / math.sqrt(n)).to(device=device, dtype=dtype)
    _H_CACHE[key] = H
    return H


def rot(x, block):
    """
    Block-diagonal rotation along the last dim. Self-inverse.

    Activations want H x and weights want W H^T; written row-wise
    both are (row @ H), since H is symmetric.
    """
    K = x.shape[-1]
    if K % block:
        raise ValueError(f"K={K} not divisible by rotation block {block}")
    H = get_H(block, x.device, x.dtype)
    lead = x.shape[:-1]
    return (x.reshape(*lead, K // block, block) @ H).reshape(*lead, K)


def requant_rows(x):
    """
    Symmetric per-row re-quantization to int8: the row's largest
    magnitude maps to 127.

    Returns (integers, rescale factor). Dividing the integers by
    the factor returns them to the incoming scale.
    """
    r = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.clamp(torch.round(x * r), -128.0, 127.0), r


# ============================================================
# ACTIVATION HOOK
#
# Replaces the harness's wrapped_round_to_quantized_type_args.
# Capture gathers LZ statistics on rotated, re-quantized
# integers; apply performs the full round trip.
#
# The harness's patched_quant_path() looks the wrapper up as a
# module global at call time, so rebinding that name redirects
# every call site without touching the harness file.
# ============================================================

def make_hook(h, cfg):

    orig_round = h.orig_round_helpers

    def wrapped(tensor, args, *pos_args, **kwargs):
        result = orig_round(tensor, args, *pos_args, **kwargs)
        h.PATCH_HITS["round"] += 1

        if not (h.MODE["capture"] or h.MODE["apply"]):
            return result
        if not h._is_a8_dynamic_token(args):
            return result

        mid = h.current_context["module_id"]
        if mid is None or h.current_context["base_name"] != "input":
            return result
        if not isinstance(result, torch.Tensor) or result.numel() == 0:
            return result

        block = cfg["block"]
        K = result.shape[-1]
        if K % block:
            # Cannot rotate this tensor. Leave it native; calibration
            # will skip it rather than mixing rotated and unrotated
            # tiles in one statistic.
            return result

        with torch.no_grad():
            rows = result.detach().reshape(-1, K).float()
            qr, r = requant_rows(rot(rows, block))

            if h.MODE["capture"]:
                q16 = qr.to(torch.int16)
                lz = h.compute_elementwise_lz(q16)
                colsum = lz.sum(dim=0, dtype=torch.float64)
                colmax = q16.abs().amax(dim=0).to(torch.int16)
                colsum_val = q16.sum(dim=0, dtype=torch.float64)
                hist = h.lz_hist(q16)

                prev = h.ACT_LZ_COLSUM.get(mid)
                if prev is None:
                    h.ACT_LZ_COLSUM[mid] = colsum
                    h.ACT_ROWS[mid] = rows.shape[0]
                    h.ACT_ABSMAX[mid] = colmax
                    h.ACT_VALSUM[mid] = colsum_val
                    h.ACT_LZ_HIST[mid] = hist
                    # The rescale clamps symmetrically to +-127, so the
                    # asymmetric -128 code never appears in the rotated
                    # domain. Recorded as zero rather than left unset,
                    # because the harness sums this dict.
                    h.ACT_NEG128[mid] = 0
                else:
                    if prev.numel() != colsum.numel():
                        raise RuntimeError(
                            f"Activation K changed across trials for module "
                            f"{mid}: {prev.numel()} -> {colsum.numel()}"
                        )
                    prev += colsum
                    h.ACT_ROWS[mid] += rows.shape[0]
                    h.ACT_ABSMAX[mid] = torch.maximum(h.ACT_ABSMAX[mid], colmax)
                    h.ACT_VALSUM[mid] += colsum_val
                    h.ACT_LZ_HIST[mid] += hist

                del lz, colsum, colmax, colsum_val, hist, q16
                return result

            # ---- apply mode ----
            params = h.ACT_BOUNDS.get(mid)
            if params is None:
                return result
            if K != params["lo"].numel():
                raise RuntimeError(
                    f"Activation K={K} does not match frozen quantizer "
                    f"K={params['lo'].numel()} for module id {mid}."
                )

            out = h.apply_tile_quant(
                qr, params, count_clip=h.ACT_CLIP_STATS[mid],
                dither=h.DITHER_CFG["activations"], k_tile=h.DITHER_CFG["k_tile"],
            )

            e = h.ACT_ERR[mid]
            diff = out - qr
            e[0] += float((diff ** 2).sum().item())
            e[1] += float((qr ** 2).sum().item())
            e[2] = max(e[2], float(diff.abs().max().item()))
            del diff

            back = rot(out / r, block)
            return torch.clamp(torch.round(back), -128.0, 127.0).to(result.dtype)

    return wrapped


# ============================================================
# EVALUATOR
# ============================================================

def make_evaluator_class(h, cfg):

    class RotEvaluator(h.WxAErrorEvaluator):

        def calibrate(self, num_trials, seq_len):
            """
            Rotate every weight tensor once, then let the parent
            gather statistics against the rotated integers by
            temporarily standing them in for self.native_w8. The
            originals are kept for restore_native_weights(), so the
            baseline stays untouched.
            """
            block = cfg["block"]
            print(f"\nRotating weights (block={block}) ...")
            self.w_rot_int, self.w_rot_r = {}, {}
            skipped = 0
            for name, q in self.native_w8.items():
                if q.shape[1] % block:
                    skipped += 1
                    continue
                qr, r = requant_rows(rot(q.float(), block))
                self.w_rot_int[name] = qr.to(torch.int8)
                self.w_rot_r[name] = r
            print(f"  rotated {len(self.w_rot_int)} tensor(s)"
                  + (f", skipped {skipped} (K not divisible)" if skipped else ""))
            if not self.w_rot_int:
                raise RuntimeError(
                    f"No weight tensor has K divisible by {block}."
                )

            original = self.native_w8
            self.native_w8 = self.w_rot_int
            try:
                super().calibrate(num_trials, seq_len)
            finally:
                self.native_w8 = original

        def apply_weight_bounds(self, params_by_name):
            """
            Quantize in the rotated basis, undo the per-row rescale,
            rotate back, and land on int8 at the original
            per-channel scale.
            """
            block = cfg["block"]
            n_clipped = n_total = 0
            self.weight_err = {}

            for name, module in self.quant_modules:
                params = params_by_name.get(name)
                if params is None or name not in self.w_rot_int:
                    module.weight.data.copy_(
                        self._reconstruct(name).to(module.weight.device))
                    continue

                per = [0, 0]
                qr = self.w_rot_int[name].to(torch.float32)
                out = h.apply_tile_quant(
                    qr, params, count_clip=per,
                    dither=h.DITHER_CFG["weights"], k_tile=h.DITHER_CFG["k_tile"],
                )
                n_clipped += per[0]
                n_total += per[1]

                # Error is measured in the ROTATED basis, where the
                # quantization actually happened. Measuring after the
                # round trip would fold in the final int8 rounding and
                # understate the scheme's own SNR.
                d = out - qr
                sq_err = float((d ** 2).sum().item())
                sq_sig = float((qr ** 2).sum().item())
                self.weight_err[name] = {
                    "clip_rate": per[0] / per[1] if per[1] else 0.0,
                    "rmse": math.sqrt(sq_err / max(d.numel(), 1)),
                    "max_err": float(d.abs().max().item()),
                    "snr_db": (10.0 * math.log10(sq_sig / sq_err))
                    if sq_err > 0 else float("inf"),
                }

                back = rot(out / self.w_rot_r[name], block)
                q_int = torch.clamp(torch.round(back), -128.0, 127.0)
                self._write_back(name, module, q_int)

            return n_clipped, n_total

    return RotEvaluator


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Block-Hadamard rotation + per-tile LZ quantization.")
    p.add_argument("--harness", default="wxax.py")
    p.add_argument("--rot-block", type=int, default=16,
                   help="Hadamard block size along K (power of two). Larger "
                        "blocks flatten harder but the gain saturates; 16 is "
                        "also the cheapest in hardware.")
    p.add_argument("--no-rotate", action="store_true",
                   help="Identity rotation (block 1). Keeps the per-row "
                        "re-quantization and removes the Hadamard, which "
                        "isolates how much of the gain is each.")

    p.add_argument("--models", nargs="+",
                   default=["RedHatAI/Qwen2.5-7B-quantized.w8a8"])
    p.add_argument("--k-sizes", type=int, nargs="+", default=[16])
    p.add_argument("--arms", nargs="+", default=["baseline", "both"])
    p.add_argument("--bit-stat", choices=["mean", "min", "percentile"],
                   default="mean")
    p.add_argument("--bit-percentile", type=float, default=99.0)
    p.add_argument("--sim-mode", default="shift",
                   choices=["truncate", "shift", "rescale"])
    p.add_argument("--pair-mode", choices=["bucket", "exact"], default="bucket")
    p.add_argument("--bucket-bits", type=int, nargs="+", default=None)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--eval-seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--calib-preset", choices=["wikitext", "multi"],
                   default="wikitext")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    h = load_harness(args.harness)
    if args.bucket_bits:
        h.set_bucket_bits(args.bucket_bits)

    block = 1 if args.no_rotate else args.rot_block
    cfg = {"block": block}

    h.wrapped_round_to_quantized_type_args = make_hook(h, cfg)
    RotEvaluator = make_evaluator_class(h, cfg)

    domains = (h.WIKITEXT_ONLY_DOMAINS if args.calib_preset == "wikitext"
               else h.DEFAULT_CALIBRATION_DOMAINS)

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.bit_stat
    if args.bit_stat == "percentile":
        tag += f"-p{args.bit_percentile:g}"
    tag += f"_{args.sim_mode}"
    tag += f"_bk{''.join(str(b) for b in h.SUPPORTED_BUCKET_BITS)}"
    tag += f"_rot{block}"
    log_path = os.path.join(args.results_dir, f"rot_{tag}_{ts}.txt")

    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = h.Tee(sys.__stdout__, log_file)

    rows = []

    try:
        print(f"Log: {log_path}")
        print(f"ROTATION    block-diagonal Hadamard, block={block}"
              + ("  (IDENTITY -- rescale only)" if block == 1 else ""))
        print(f"k_sizes     {args.k_sizes}")
        print(f"bit_stat    {args.bit_stat}"
              + (f" p{args.bit_percentile:g}" if args.bit_stat == "percentile" else ""))
        print(f"sim_mode    {args.sim_mode}")
        print(f"buckets     {h.SUPPORTED_BUCKET_BITS}")
        print("  Statistics are gathered on rotated, re-quantized integers.")
        print("  eff_bits is expected to RISE relative to an unrotated run;")
        print("  the payoff is in mean shift depth and therefore in PPL.")
        print("NOTE: calibration is WikiText-2 train against WikiText-2 test, "
              "so the deltas are mildly optimistic.")

        for idx, model_name in enumerate(args.models, 1):
            print("\n" + "#" * 90)
            print(f"##  {idx}/{len(args.models)}  {model_name}   [rot {block}]")
            print("#" * 90 + "\n")

            ev = None
            try:
                ev = RotEvaluator(
                    model_name, args.k_sizes, domains,
                    pair_mode=args.pair_mode, sim_mode=args.sim_mode,
                    bit_stat=args.bit_stat, bit_percentile=args.bit_percentile,
                )
                ev.calibrate(args.trials, args.calib_seq_len)
                ev.run(args.eval_seq_len, args.max_windows, args.arms, rows)
            except Exception as e:
                print(f"\n!!! ERROR on {model_name}: {e}")
                traceback.print_exc()
            finally:
                if ev is not None:
                    del ev
                for d in (h.ACT_LZ_COLSUM, h.ACT_ROWS, h.ACT_ABSMAX,
                          h.ACT_VALSUM, h.ACT_LZ_HIST, h.ACT_NEG128,
                          h.ACT_BOUNDS, h.ACT_CLIP_STATS, h.ACT_ERR):
                    d.clear()
                _H_CACHE.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        if rows:
            print("\n" + "=" * 90)
            print("SUMMARY")
            print("=" * 90)
            hdr = (f"{'k_tile':>7s} {'arm':<10s} {'eff bits':>9s} {'shift':>7s} "
                   f"{'PPL':>12s} {'dPPL':>12s} {'%':>10s}")
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                print(f"{str(r['k_tile']):>7s} {r['arm']:<10s} "
                      f"{r['eff_bits']:>9s} {r.get('mean_shift', '-'):>7s} "
                      f"{float(r['ppl']):12.4f} {r['delta_ppl'] or '-':>12s} "
                      f"{(r['pct_delta'] + '%') if r['pct_delta'] else '-':>10s}")
            print("-" * len(hdr))
            print(f"config: rotation block={block}, bit_stat={args.bit_stat}"
                  + (f" p{args.bit_percentile:g}" if args.bit_stat == "percentile" else "")
                  + f", sim_mode={args.sim_mode}"
                  + f", buckets={h.SUPPORTED_BUCKET_BITS}"
                  + f", calib={args.calib_preset} x{args.trials} @{args.calib_seq_len}"
                  + f", eval @{args.eval_seq_len}")

    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log written: {log_path}")


if __name__ == "__main__":
    main()