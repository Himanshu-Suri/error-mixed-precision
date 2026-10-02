#!/usr/bin/env python3
"""
============================================================
MULTI-MODEL RUN + PRECISION CENSUS
============================================================

Runs the error harness across several models and, for each one,
prints how much of the model actually landed in each precision
mode -- counted three ways, because they do not agree and the
difference is the point.

WHY TWO COUNTS
--------------
The harness reports the share of TILES per bucket. That
over-represents narrow tensors: a K-tile of k_proj and a K-tile
of down_proj both count as one, but the second covers far more
parameters. So this adds:

  Tiles %     share of K-tiles. What the harness prints.
  Weights %   share of weight PARAMETERS, = k_tile * M per tile.
              The memory-footprint answer: "what fraction of the
              weights are stored at b bits?"

A scheme can look good on tiles and poor on parameters, which is
the case worth catching before reporting a result. Each census
ends with the fraction below 8 bits on both measures, and quoting
one without saying which is how two papers end up disagreeing
about the same scheme.

WHY NOT ACTIVATION OR MAC SHARES
--------------------------------
An earlier version of this script also reported activation-element
and MAC shares. Both are degenerate here. Activation elements per
tile are k_tile * N, and N -- the pooled calibration row count --
is identical for every tensor in a dense decoder, so that column
is a constant multiple of the tile count. MACs per tile are
k_tile * M * N, and the constant N cancels in the ratio, so MAC
shares reduce exactly to parameter shares. Three of the four
columns carried two numbers. They would separate on MoE or any
architecture where modules see different row counts.

USAGE
-----
  python multi_census.py --harness wxa_error.py \\
      --models A B C --bit-percentile 90
============================================================
"""

import argparse
import gc
import importlib.util
import os
import sys
import traceback
from collections import Counter
from datetime import datetime

import torch


def load_harness(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"harness not found: {path}")
    spec = importlib.util.spec_from_file_location("_wxa_err", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wxa_err"] = mod
    spec.loader.exec_module(mod)
    for need in ("WxAErrorEvaluator", "SUPPORTED_BUCKET_BITS", "set_bucket_bits",
                 "WIKITEXT_ONLY_DOMAINS", "DEFAULT_CALIBRATION_DOMAINS", "Tee"):
        if not hasattr(mod, need):
            raise RuntimeError(
                f"'{os.path.basename(path)}' has no {need}. This runner needs the "
                f"ERROR harness (WxAErrorEvaluator / eval_wikitext_ppl), not the "
                f"distribution analysis script (WxAPairAnalyzer)."
            )
    return mod


# ============================================================
# CENSUS
# ============================================================

def precision_census(h, ev, k_tile):
    """
    Walk the frozen bit-vectors and total up, per bucket, the
    tiles / weight parameters / activation elements / MACs that
    landed there.

    Everything is read from what calibration already froze --
    frozen_bits and shape_info -- so this adds no forward passes
    and cannot disagree with the bit-widths the run actually uses.
    """
    frozen = ev.frozen_bits.get(k_tile)
    if not frozen:
        print(f"  k_tile={k_tile}: nothing frozen")
        return None

    tiles, params = Counter(), Counter()

    for name, bits in frozen.items():
        M, K, N = ev.shape_info[name]
        per_tile_params = float(k_tile) * float(M)
        for b in bits["bucket"]:
            tiles[b] += 1
            params[b] += per_tile_params

    tot_t = sum(tiles.values())
    tot_p = sum(params.values())
    if tot_t == 0:
        return None

    print(f"\n  PRECISION CENSUS  (k_tile={k_tile})")
    print(f"    {'mode':>6s} {'tiles':>10s} {'Tiles %':>9s} {'Weights %':>10s}")
    print("    " + "-" * 39)
    for b in sorted(set(list(tiles.keys()) + list(h.SUPPORTED_BUCKET_BITS))):
        t = tiles.get(b, 0)
        print(f"    {('W%dA%d' % (b, b)):>6s} {t:10d} "
              f"{t / tot_t * 100:8.2f}% "
              f"{params.get(b, 0.0) / tot_p * 100:9.2f}%")
    print("    " + "-" * 39)

    low_t = sum(c for b, c in tiles.items() if b < 8)
    low_p = sum(c for b, c in params.items() if b < 8)
    print(f"    {'< 8b':>6s} {low_t:10d} "
          f"{low_t / tot_t * 100:8.2f}% {low_p / tot_p * 100:9.2f}%")

    mean_param_bits = sum(b * c for b, c in params.items()) / tot_p
    print(f"    parameter-weighted mean width: {mean_param_bits:.3f} bits")
    if abs(low_t / tot_t - low_p / tot_p) * 100 > 3.0:
        print("    Tiles % and Weights % diverge by more than 3 points: the")
        print("    sub-8-bit tiles sit disproportionately in narrow tensors, so")
        print("    the tile count overstates how much of the model is compressed.")

    return {
        "tiles_total": tot_t,
        "low_tiles_pct": low_t / tot_t * 100,
        "low_params_pct": low_p / tot_p * 100,
        "param_bits": mean_param_bits,
        "dist": dict(tiles),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--harness", default="wxa_error.py")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--k-tile", type=int, default=16)
    p.add_argument("--bit-stat", default="percentile",
                   choices=["mean", "min", "percentile"])
    p.add_argument("--bit-percentile", type=float, default=90.0)
    p.add_argument("--sim-mode", default="shift", choices=["truncate", "shift", "rescale"])
    p.add_argument("--bucket-bits", type=int, nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=["baseline", "both"])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--eval-seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap eval windows. Use a small value first: this runner "
                        "loads every model in turn, so a late failure is expensive.")
    p.add_argument("--calib-preset", choices=["wikitext", "multi"], default="wikitext")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--census-only", action="store_true",
                   help="Calibrate and print the census, skip the perplexity "
                        "evaluation. Minutes per model instead of an hour.")
    args = p.parse_args()

    h = load_harness(args.harness)
    if args.bucket_bits:
        h.set_bucket_bits(args.bucket_bits)

    domains = (h.WIKITEXT_ONLY_DOMAINS if args.calib_preset == "wikitext"
               else h.DEFAULT_CALIBRATION_DOMAINS)

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.bit_stat
    if args.bit_stat == "percentile":
        tag += f"-p{args.bit_percentile:g}"
    log_path = os.path.join(args.results_dir, f"census_{tag}_{args.sim_mode}_{ts}.txt")

    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = h.Tee(sys.__stdout__, log_file)

    summary = []

    try:
        print(f"Log: {log_path}")
        print(f"models      {len(args.models)}")
        print(f"k_tile      {args.k_tile}")
        print(f"bit_stat    {args.bit_stat}"
              + (f" p{args.bit_percentile:g}" if args.bit_stat == "percentile" else ""))
        print(f"sim_mode    {args.sim_mode}")
        print(f"buckets     {h.SUPPORTED_BUCKET_BITS}")
        print("NOTE: calibration is WikiText-2 train against WikiText-2 test, so "
              "the deltas are mildly optimistic.")

        for idx, model_name in enumerate(args.models, 1):
            print("\n" + "#" * 90)
            print(f"##  {idx}/{len(args.models)}  {model_name}")
            print("#" * 90 + "\n")

            ev, rows = None, []
            try:
                ev = h.WxAErrorEvaluator(
                    model_name, [args.k_tile], domains,
                    pair_mode="bucket", sim_mode=args.sim_mode,
                    bit_stat=args.bit_stat, bit_percentile=args.bit_percentile,
                )
                ev.calibrate(args.trials, args.calib_seq_len)

                census = precision_census(h, ev, args.k_tile)
                eff_bits, coverage = ev.compute_weighted_effective_bits(args.k_tile)

                base = quant = None
                if not args.census_only:
                    ev.run(args.eval_seq_len, args.max_windows, args.arms, rows)
                    for r in rows:
                        if r["arm"] == "baseline":
                            base = float(r["ppl"])
                        elif r["arm"] == "both":
                            quant = float(r["ppl"])

                if census:
                    summary.append({
                        "model": model_name.split("/")[-1],
                        "eff_bits": eff_bits,
                        "param_bits": census["param_bits"],
                        "shift": getattr(ev, "_mean_shift", float("nan")),
                        "base": base, "ppl": quant,
                        "low_t": census["low_tiles_pct"],
                        "low_p": census["low_params_pct"],
                    })

            except Exception as e:
                print(f"\n!!! ERROR on {model_name}: {e}")
                traceback.print_exc()
            finally:
                if ev is not None:
                    del ev
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # ---- cross-model table ----
        if summary:
            print("\n" + "=" * 110)
            print("CROSS-MODEL SUMMARY")
            print("=" * 110)
            hdr = (f"{'model':<34s} {'bits':>6s} {'shift':>6s} "
                   f"{'base':>8s} {'PPL':>9s} {'dPPL %':>8s} "
                   f"{'<8b tiles':>10s} {'<8b wts':>9s}")
            print(hdr)
            print("-" * len(hdr))
            for s in summary:
                pct = (f"{(s['ppl'] - s['base']) / s['base'] * 100:+7.2f}%"
                       if s["base"] and s["ppl"] else "      -")
                base_s = f"{s['base']:8.4f}" if s["base"] else "       -"
                ppl_s = f"{s['ppl']:9.4f}" if s["ppl"] else "        -"
                print(f"{s['model'][:34]:<34s} {s['eff_bits']:6.3f} "
                      f"{s['shift']:6.3f} {base_s} {ppl_s} {pct:>8s} "
                      f"{s['low_t']:9.1f}% {s['low_p']:8.1f}%")
            print("-" * len(hdr))
            print("  bits   = mean operand width, MAC-weighted (matches the harness).")
            print("  <8b    = share of K-tiles, and of weight parameters, assigned")
            print("           below 8 bits. The two differ when the sub-8-bit tiles")
            print("           sit in narrow tensors, which is the case worth noticing.")
            print("  Activation-element and MAC shares are omitted: on a dense decoder")
            print("  every tensor sees the same calibration row count, so both collapse")
            print("  onto the tile and parameter columns and carry no extra information.")

    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log written: {log_path}")


if __name__ == "__main__":
    main()