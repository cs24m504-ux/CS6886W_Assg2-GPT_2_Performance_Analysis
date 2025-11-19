#!/usr/bin/env python3
"""
oi_perf.py — Compute Operation Intensity (FLOPs / Byte) using `perf stat`
Single execution: runs your workload once and collects all counters together.

Usage:
  sudo python3 oi_perf.py [--assume-fma] [--add-ld DIR ...] -- <command> [args...]

Options:
  --assume-fma              Count packed FP ops as FMAs (×2 FLOPs per element)
  --add-ld DIR              Prepend DIR to LD_LIBRARY_PATH (repeatable)
  --debug                   Print diagnostic info
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict

# ------------------------- Event configuration -------------------------

FP_EVENTS = [
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.scalar_double",
    "fp_arith_inst_retired.128b_packed_single",
    "fp_arith_inst_retired.128b_packed_double",
    "fp_arith_inst_retired.256b_packed_single",
    "fp_arith_inst_retired.256b_packed_double",
]

IMC_READ  = "uncore_imc_free_running/data_read/"
IMC_WRITE = "uncore_imc_free_running/data_write/"
IMC_TOTAL = "uncore_imc_free_running/data_total/"

FALLBACK_LOADS  = "mem_inst_retired.all_loads"
FALLBACK_STORES = "mem_inst_retired.all_stores"

ALL_EVENTS = FP_EVENTS + [IMC_READ, IMC_WRITE, IMC_TOTAL, FALLBACK_LOADS, FALLBACK_STORES]

# ------------------------------- Helpers -------------------------------

def build_ld_library_path(extra_dirs):
    """
    Assemble LD_LIBRARY_PATH:
    - auto-detect common oneAPI locations
    - include user-provided --add-ld dirs
    - preserve existing LD_LIBRARY_PATH
    """
    candidates = [
        "/opt/intel/oneapi/2025.3/lib",
        "/opt/intel/oneapi/installer/lib",
        "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin",
    ]
    root = "/opt/intel/oneapi"
    if os.path.isdir(root):
        for sub in os.listdir(root):
            p = os.path.join(root, sub, "lib")
            if os.path.isdir(p):
                candidates.append(p)
            p2 = os.path.join(root, "compiler", sub, "linux", "compiler", "lib", "intel64_lin")
            if os.path.isdir(p2):
                candidates.append(p2)

    seen, parts = set(), []
    for d in candidates + list(extra_dirs):
        if os.path.isdir(d) and d not in seen:
            parts.append(d)
            seen.add(d)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current:
        for d in current.split(":"):
            if d and d not in seen:
                parts.append(d)
                seen.add(d)

    return ":".join(parts)

def to_bytes(value_str, unit):
    """Convert perf's numeric value + unit into bytes (for IMC counters)."""
    if not value_str or value_str.startswith("<"):
        return 0.0
    try:
        val = float(value_str)
    except ValueError:
        return 0.0
    u = (unit or "").strip()
    if u in ("", "counts", "insn"):
        return val
    if u in ("B", "bytes"):
        return val
    if u == "KiB":
        return val * 1024
    if u == "MiB":
        return val * 1048576
    if u == "GiB":
        return val * 1073741824
    if u == "TiB":
        return val * 1099511627776
    if u == "kB":
        return val * 1000
    if u == "MB":
        return val * 1_000_000
    if u == "GB":
        return val * 1_000_000_000
    if u == "TB":
        return val * 1_000_000_000_000
    return val

def normalize_event_name(event_field):
    """
    Normalize the perf event string so lookups work regardless of PMU prefix/suffix:
      cpu_core/fp_arith_inst_retired.scalar_single/  -> fp_arith_inst_retired.scalar_single
      cpu_atom/…/                                     -> …
      uncore_imc_free_running/data_read/              -> unmodified except trailing slash removed
    """
    e = event_field.strip().lower()
    if e.startswith("cpu_core/"):
        e = e[len("cpu_core/"):]
    elif e.startswith("cpu_atom/"):
        e = e[len("cpu_atom/"):]
    if e.endswith("/"):
        e = e[:-1]
    return e

def run_perf_and_parse(cmd, events, ld_library_path, debug=False):
    """
    Run one perf stat call and parse CSV stderr lines.
    Returns: { normalized_event_name: (value_str, unit_str) }
    """
    perf_cmd = [
        "perf", "stat",
        "-x", ",",
        "--no-big-num",
        "-a",
        "-e", ",".join(events),
        "--", "env", f"LD_LIBRARY_PATH={ld_library_path}",
    ] + cmd

    print("Running perf command: \n", " ".join(shlex.quote(x) for x in perf_cmd), "\n")

    proc = subprocess.run(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = proc.stderr.splitlines()

    results = defaultdict(list)

    for line in stderr:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        val_str, unit, event = parts[0], parts[1], parts[2]
        if not event or val_str.startswith("<"):
            continue
        norm = normalize_event_name(event)
        results[norm].append((val_str, unit))

    # Last occurrence per event (aggregated)
    final_map = {k: vlist[-1] for k, vlist in results.items()}
    return final_map

def get_count(final_map, event_name):
    target = normalize_event_name(event_name)
    tup = final_map.get(target)
    if not tup:
        return 0.0
    val_str, _unit = tup
    try:
        return float(val_str) if val_str and not val_str.startswith("<") else 0.0
    except ValueError:
        return 0.0

def get_bytes_from(final_map, event_name):
    target = normalize_event_name(event_name)
    tup = final_map.get(target)
    if not tup:
        return 0.0
    val_str, unit = tup
    return to_bytes(val_str, unit)

# --------------------------------- Main ---------------------------------

def main():
    p = argparse.ArgumentParser(description="Compute Operation Intensity (FLOPs/Byte) using perf")
    p.add_argument("--assume-fma", action="store_true",
                   help="Assume packed FP ops are FMAs (×2 FLOPs per element)")
    p.add_argument("--add-ld", action="append", default=[],
                   help="Extra directory to prepend to LD_LIBRARY_PATH (repeatable)")
    p.add_argument("--debug", action="store_true", help="Print diagnostics")
    known, remaining = p.parse_known_args()
    args = known  # alias

    # Robust handling of separator: if '--' exists, use everything after it; else use remaining.
    if "--" in sys.argv[1:]:
        idx = sys.argv[1:].index("--") + 1  # offset from prog name
        cmd = sys.argv[idx + 1:]
    else:
        cmd = remaining

    if not cmd:
        print("Usage: sudo python3 oi_perf.py [--assume-fma] [--add-ld DIR ...] -- <command> [args...]", file=sys.stderr)
        sys.exit(1)

    ld_final = build_ld_library_path(args.add_ld)

    if args.debug:
        print("DEBUG argv:", sys.argv[1:])
        print("DEBUG known:", known)
        print("DEBUG cmd:", cmd)
        print("DEBUG LD_LIBRARY_PATH:", ld_final or "<empty>")

    # Measure wall-clock time around the perf run
    t0 = time.perf_counter()
    final_map = run_perf_and_parse(cmd, ALL_EVENTS, ld_final, debug=args.debug)
    elapsed_sec = time.perf_counter() - t0

    # FP counts
    s_scalar_s = get_count(final_map, FP_EVENTS[0])
    s_scalar_d = get_count(final_map, FP_EVENTS[1])
    s_128_s    = get_count(final_map, FP_EVENTS[2])
    s_128_d    = get_count(final_map, FP_EVENTS[3])
    s_256_s    = get_count(final_map, FP_EVENTS[4])
    s_256_d    = get_count(final_map, FP_EVENTS[5])

    # Bytes (prefer R+W, then total, else fallback)
    bytes_rd = get_bytes_from(final_map, IMC_READ)
    bytes_wr = get_bytes_from(final_map, IMC_WRITE)
    total_bytes = bytes_rd + bytes_wr
    bytes_src = "IMC(read+write)"
    approx_note = ""

    if round(total_bytes) == 0:
        total_bytes = get_bytes_from(final_map, IMC_TOTAL)
        bytes_src = "IMC(total)"

    if round(total_bytes) == 0:
        loads  = get_count(final_map, FALLBACK_LOADS)
        stores = get_count(final_map, FALLBACK_STORES)
        total_bytes = (loads + stores) * 64.0
        bytes_src = "In-core proxy (64B × (loads+stores))"
        approx_note = "(approximate; not DRAM-specific)"

    # FLOPs
    fma_factor = 2.0 if args.assume_fma else 1.0
    flops = (
        s_scalar_s + s_scalar_d +
        (s_128_s + s_128_d) * 2.0 * fma_factor +
        (s_256_s + s_256_d) * 4.0 * fma_factor
    )

    # OI and throughput
    oi = float("nan")
    if round(total_bytes) != 0:
        oi = flops / total_bytes

    gflops_per_s = float("nan")
    gbytes_per_s = float("nan")
    if elapsed_sec > 0.0:
        if flops > 0.0:
            gflops_per_s = flops / (1e9 * elapsed_sec)
        if total_bytes > 0.0:
            gbytes_per_s = total_bytes / (1e9 * elapsed_sec)

    # Pretty printing helpers
    def fmt(x):
        if isinstance(x, float):
            return f"{x:,.6f}"
        return f"{x:,}"

    def fmi(x):
        return f"{int(x):,}"

    # Report
    print("=============== Operation Intensity Report ===============")
    print("Command:", " ".join(shlex.quote(c) for c in cmd))
    print("Injected LD_LIBRARY_PATH:")
    print(" ", ld_final if ld_final else "<empty>")
    print()
    print(f"Assume FMA for packed ops: {'YES (×2 per element)' if args.assume_fma else 'NO'}")
    print()
    print("FLOP counters (retired):")
    print(f"  {FP_EVENTS[0]:45s} {fmi(s_scalar_s)}")
    print(f"  {FP_EVENTS[1]:45s} {fmi(s_scalar_d)}")
    print(f"  {FP_EVENTS[2]:45s} {fmi(s_128_s)}")
    print(f"  {FP_EVENTS[3]:45s} {fmi(s_128_d)}")
    print(f"  {FP_EVENTS[4]:45s} {fmi(s_256_s)}")
    print(f"  {FP_EVENTS[5]:45s} {fmi(s_256_d)}")
    print()
    print(f"Memory traffic source: {bytes_src} {approx_note}")
    if bytes_src == "IMC(read+write)":
        print(f"  {IMC_READ:45s} {fmt(bytes_rd)} bytes")
        print(f"  {IMC_WRITE:45s} {fmt(bytes_wr)} bytes")
    print(f"  {'Total Bytes':45s} {fmt(total_bytes)} bytes")
    print()
    print("Totals:")
    print(f"  FLOPs      = {fmt(flops)}")
    print(f"  Bytes      = {fmt(total_bytes)}")
    print(f"  OI         = FLOPs / Bytes = {fmt(oi)} FLOPs/Byte")
    print(f"  Time       = {elapsed_sec:.6f} seconds (wall clock, including perf)")
    if flops > 0.0:
        print(f"  GFLOPs/s   = {gflops_per_s:.6f}")
    if total_bytes > 0.0:
        print(f"  GB/s       = {gbytes_per_s:.6f}")
    print("==========================================================")

if __name__ == "__main__":
    main()
