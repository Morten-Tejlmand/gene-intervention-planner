# run all 4 experiments in sequence
# use --skip-slow to skip exp03 and exp04 (the AL simulations take a while)
#
# Exp01 - baseline model comparison (LR / RF / GBT cross-validation)
# Exp02 - feature set comparison (biophysics -> +AA -> +evolutionary -> +GO/ESM-2)
# Exp03 - AL vs random (random / margin / UCB on full universe)
# Exp04 - hybrid acquisition strategies (random / margin / UCB)

import subprocess
import sys
import time

SKIP_SLOW = "--skip-slow" in sys.argv

EXPERIMENTS = [
    ("experiments/behavioural_al/exp01_baseline.py",        "Exp01  Baseline model comparison",      False),
    ("experiments/behavioural_al/exp02_feature_comparison.py", "Exp02  Feature set comparison",      False),
    ("experiments/behavioural_al/exp03_al_vs_random.py",    "Exp03  AL vs random (3 strategies)",    True),
    ("experiments/behavioural_al/exp04_hybrid.py",          "Exp04  Hybrid acquisition strategies",  True),
]

results = []

for script, label, is_slow in EXPERIMENTS:
    if is_slow and SKIP_SLOW:
        print(f"  SKIP  {label}")
        results.append((label, "skipped", 0))
        continue

    print(f"\n{'=' * 60}")
    print(f"  RUN   {label}")
    print("=" * 60)

    t0 = time.time()
    ret = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0

    status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
    results.append((label, status, elapsed))
    print(f"  -> {status}  ({elapsed:.0f}s)")

print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
for label, status, elapsed in results:
    tag = "OK" if status == "OK" else ("SKIP" if status == "skipped" else "FAIL")
    if elapsed:
        print(f"  [{tag}]  {label:<45}  {elapsed:.0f}s")
    else:
        print(f"  [{tag}]  {label}")

print("\nDone. Artifacts in artifacts/behavioural_al/")
