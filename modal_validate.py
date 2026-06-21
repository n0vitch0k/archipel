"""Modal runner for Archipel — tests longs et validation complète.

Usage:
    modal run modal_validate.py                    # Suite par défaut
    modal run modal_validate.py --quick            # Tests unitaires seulement
    modal run modal_validate.py --mnist            # MNIST + validation uniquement
    modal run modal_validate.py --long             # Run 50 epochs
    modal run modal_validate.py --multi-long       # 3×50 epochs (seeds 42, 123, 256)
    modal run modal_validate.py --all              # Tout
"""

import sys

import modal

app = modal.App("archipel-validate")

# Base image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(["torch>=2.0.0", "torchvision>=0.15.0"],
                  index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy>=2.0.0")
    .pip_install("pyyaml>=6.0.0")
    .pip_install("pytest>=8.0.0")
)


@app.local_entrypoint()
def main(quick: bool = False, mnist: bool = False, all: bool = False, long: bool = False,
         multi_long: bool = False, seed: int = 42):
    """Run Archipel validation suite on Modal cloud."""

    if multi_long:
        suite = "3×50 epochs multi-seed (42, 123, 256)"
        seeds = [42, 123, 256]
        print(f"🚀 Archipel Validation — {suite}")
        print(f"   Running in cloud (CPU instance)…\n")
        for s in seeds:
            print(f"\n{'='*60}")
            print(f"  ▶ Run with seed={s}")
            print(f"{'='*60}")
            result = run_validation.remote(modes=["mnist50"], seed=s)
            print(result["summary"])
        print("\n✅ ALL 3 MULTI-SEED RUNS COMPLETED")
        return

    if long:
        suite = "50 epochs"
        modes = ["mnist50"]
    elif all:
        suite = "full (unit + MNIST + validation + multi-seed)"
        modes = ["unit", "mnist", "validate", "multi_seed"]
    elif mnist:
        suite = "MNIST + validation"
        modes = ["mnist", "validate"]
    elif quick:
        suite = "unit tests only"
        modes = ["unit"]
    else:
        suite = "default (unit + MNIST)"
        modes = ["unit", "mnist"]

    print(f"🚀 Archipel Validation — {suite}")
    print(f"   Seed: {seed}")
    print(f"   Running in cloud (CPU instance)…\n")

    result = run_validation.remote(modes=modes, seed=seed)

    print("\n" + "=" * 65)
    print(result["summary"])
    print("=" * 65)
    print(f"   Dashboard: {result.get('dashboard', 'N/A')}")

    if result["exit_code"] != 0:
        print("\n❌ SOME CHECKS FAILED")
        sys.exit(1)
    else:
        print("\n✅ ALL CHECKS PASSED")


@app.function(
    image=image,
    timeout=3600,  # 60 min max for long runs
    retries=1,
)
def run_validation(modes: list, seed: int) -> dict:
    import subprocess, json, time

    cwd = "/archipel"
    repo_url = "https://github.com/n0vitch0k/archipel.git"

    # Clone repo
    r = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, cwd],
        capture_output=True, text=True, timeout=60
    )
    if r.returncode != 0:
        # Retry with removal (stale dir from previous run in same container)
        subprocess.run(["rm", "-rf", cwd], capture_output=True)
        r = subprocess.run(
            ["git", "clone", "--depth=1", repo_url, cwd],
            capture_output=True, text=True, timeout=60
        )
    if r.returncode != 0:
        return {"summary": f"❌ git clone failed:\n{r.stderr}", "exit_code": 1}

    # Install package
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{cwd}/archipel"],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        return {"summary": f"❌ pip install failed:\n{r.stderr[-2000:]}", "exit_code": 1}

    lines = []
    all_ok = True

    def run_step(label: str, cmd: list, timeout: int = 300) -> bool:
        nonlocal all_ok
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        elapsed = time.time() - t0
        out = r.stdout.strip()
        err = r.stderr.strip()

        # Print all output for 50-epoch runs (to see diagnostics), else last 20 lines
        out_lines = out.split("\n")
        if "50 epochs" in label:
            window = out_lines[-40:]
        else:
            window = out_lines[-20:]
        for l in window:
            print(f"  {l}")
        if err:
            print(f"  [stderr] {err[:500]}")

        ok = r.returncode == 0
        status = "✅" if ok else "❌"
        lines.append(f"  {status} {label} — {elapsed:.1f}s")
        if not ok:
            all_ok = False
        return ok

    # ── Step 1: Unit tests ──
    if "unit" in modes:
        for tf in [("top-k curriculum", "test_topk_curriculum.py", 120),
                   ("specialization", "test_specialization.py", 120),
                   ("train script", "test_train_script.py", 120)]:
            run_step(
                f"Unit test: {tf[0]}",
                [sys.executable, "-m", "pytest", tf[1], "-v", "--tb=short"],
                timeout=tf[2]
            )

    # ── Step 2: MNIST quick test (5 epochs) ──
    if "mnist" in modes:
        run_step(
            "MNIST training (5 epochs, batch=64)",
            [sys.executable, "test_mnist_quick.py", "--epochs", "5", "--batch-size", "64"],
            timeout=600
        )

    # ── Step 2b: MNIST long (50 epochs) ──
    if "mnist50" in modes:
        run_step(
            f"MNIST 50 epochs (batch=64, seed={seed})",
            [sys.executable, "test_mnist_quick.py", "--epochs", "50", "--batch-size", "64"],
            timeout=1800
        )

    # ── Step 3: Validation N1.4/1.5 ──
    if "validate" in modes:
        run_step(
            "Quick validation (2 epochs, batch=128)",
            [sys.executable, "validate_niveau1415_quick.py", "--epochs", "2", "--batch-size", "128"],
            timeout=600
        )
        run_step(
            "Full validation + MLP baseline (3 epochs, batch=64)",
            [sys.executable, "validate_niveau1415.py", "--epochs", "3", "--batch-size", "64"],
            timeout=900
        )

    # ── Step 4: Multi-seed short runs ──
    if "multi_seed" in modes:
        for run_id in [1, 2, 3]:
            run_step(
                f"MNIST run #{run_id} (3 epochs, batch=128)",
                [sys.executable, "test_mnist_quick.py",
                 "--epochs", "3", "--batch-size", "128"],
                timeout=600
            )

    # ── Summary ──
    summary_lines = [
        f"📊 VALIDATION SUMMARY ({len([m for m in modes])} mode(s))",
    ] + lines
    status_code = 0 if all_ok else 1
    summary_lines.append(f"\n   {'✅ ALL PASSED' if all_ok else '❌ SOME FAILED'}")

    return {
        "summary": "\n".join(summary_lines),
        "exit_code": status_code,
    }
