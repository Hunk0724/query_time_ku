#!/usr/bin/env python3
"""Compare your fresh runs against the shipped reference in reference_outputs/.

Your runs write to outputs/ (FC-SH) and analysis/results/lme_hyps/ (LME-KU); the
shipped reference (the exact run products behind the paper) lives in
reference_outputs/. This script recomputes the paper metric for a fresh result
file and diffs it against the matching file in reference_outputs/, so you can
confirm a reproduction without touching the reference.

Metric:
  FC-SH  = substring exact match (sEM) = mean of the `substring_exact_match`
           field over a result file's rows (the metric in every FC-SH table).
  LME-KU = fraction of judged rows with autoeval label true, out of 78.

Match: a fresh file at outputs/<relpath> is compared to reference_outputs/<relpath>
(same relative path); a fresh LME eval-results file is matched to
reference_outputs/lme_hyps/<basename>.

A cell PASSES if within TOL percentage points of the reference (default 3, for
temperature-0 server nondeterminism).

Usage:
  python scripts/compare_to_reference.py                  # scan outputs/ + lme_hyps
  python scripts/compare_to_reference.py --fc  outputs/.../<result>.json
  python scripts/compare_to_reference.py --lme <hyp>.jsonl.eval-results-gpt-4o-mini
  python scripts/compare_to_reference.py --tol 2
"""
import argparse, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRESH = os.path.join(ROOT, "outputs")
REF = os.path.join(ROOT, "reference_outputs")
LME_FRESH = os.path.join(ROOT, "analysis", "results", "lme_hyps")
LME_REF = os.path.join(REF, "lme_hyps")


def fc_sem(path):
    rows = json.load(open(path))["data"]
    return sum(1 for r in rows if r.get("substring_exact_match")), len(rows)


def lme_count(path):
    n = t = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        lab = json.loads(line).get("autoeval_label", {})
        n += 1
        if (lab.get("label") if isinstance(lab, dict) else lab):
            t += 1
    return t, n


def ref_of(fresh_path):
    """Map a fresh path under outputs/ to its reference under reference_outputs/."""
    return os.path.join(REF, os.path.relpath(os.path.abspath(fresh_path), FRESH))


def cmp_fc(paths, tol):
    print(f"== FC-SH (sEM, tol +/-{tol}pp) ==")
    any_row = False
    for p in paths:
        refp = ref_of(p)
        if not os.path.isfile(refp):
            print(f"  [no ref]  {os.path.relpath(p, ROOT)}")
            continue
        (s, n), (rs, rn) = fc_sem(p), fc_sem(refp)
        pct, rpct = (100 * s / n if n else 0), (100 * rs / rn if rn else 0)
        ok = "PASS" if abs(pct - rpct) <= tol else "DIFF"
        any_row = True
        print(f"  [{ok}] {os.path.relpath(p, FRESH):60} you {s}/{n}={pct:.1f}  ref {rs}/{rn}={rpct:.1f}")
    if not any_row:
        print("  (no matching fresh FC-SH result files found under outputs/)")


def cmp_lme(paths, tol):
    print(f"== LME-KU (accuracy, tol +/-{tol}pp) ==")
    for p in paths:
        refp = os.path.join(LME_REF, os.path.basename(p))
        if not os.path.isfile(refp):
            print(f"  [no ref]  {os.path.basename(p)}")
            continue
        (t, n), (rt, rn) = lme_count(p), lme_count(refp)
        pct, rpct = (100 * t / n if n else 0), (100 * rt / rn if rn else 0)
        ok = "PASS" if abs(pct - rpct) <= tol else "DIFF"
        print(f"  [{ok}] {os.path.basename(p):50} you {t}/{n}={pct:.1f}  ref {rt}/{rn}={rpct:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", nargs="*", help="FC-SH result json(s)")
    ap.add_argument("--lme", nargs="*", help="LME eval-results file(s)")
    ap.add_argument("--tol", type=float, default=3.0)
    a = ap.parse_args()
    fc_in = a.fc if a.fc is not None else glob.glob(
        os.path.join(FRESH, "**", "*results*.json"), recursive=True)
    lme_in = a.lme if a.lme is not None else glob.glob(
        os.path.join(LME_FRESH, "*.eval-results-*"))
    fc = [p for p in fc_in if p and os.path.isfile(p)]
    lme = [p for p in lme_in if p and os.path.isfile(p)]

    # Paths that were named but do not resolve to a readable file. On Windows a
    # >260-char path can survive extraction yet fail os.path.isfile() — warn
    # instead of silently dropping it into a bare "Nothing to compare".
    missing = [p for p in fc_in + lme_in if p and not os.path.isfile(p)]
    if missing:
        print(f"!! {len(missing)} named path(s) are not readable and were skipped "
              f"(on Windows, likely the >260-char path limit — see README section 5):")
        for p in missing:
            print(f"   {p}")
        print()

    if fc:
        cmp_fc(fc, a.tol)
    if lme:
        cmp_lme(lme, a.tol)
    if not fc and not lme and not missing:
        print("Nothing to compare. Run some experiments first (see README section 3), "
              "or pass --fc / --lme paths.")
