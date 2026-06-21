"""Modal runner for Archipel CI tests.

Usage:
    modal run modal_run.py              # Run test suite
    modal run modal_run.py --quick      # Run only top-k + specialization tests
"""

import sys

import modal

app = modal.App("archipel-ci")

# Build a lightweight image with torch CPU + our deps + git
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
def main(quick: bool = False):
    """Run Archipel tests on Modal cloud infrastructure."""
    print(f"🚀 Archipel CI — {'quick suite' if quick else 'full suite'}")
    print(f"   Running in cloud (CPU instance)…\n")
    result = run_tests.remote(quick=quick)

    print("\n" + "=" * 60)
    print(result["summary"])
    print("=" * 60)

    if result["exit_code"] != 0:
        print("\n❌ SOME TESTS FAILED — check output above")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")


@app.function(
    image=image,
    timeout=600,
    retries=1,
)
def run_tests(quick: bool = False) -> dict:
    import subprocess

    # 1. Grab the repo
    repo_url = "https://github.com/n0vitch0k/archipel.git"

    clone_result = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, "/archipel"],
        capture_output=True, text=True, timeout=60
    )
    if clone_result.returncode != 0:
        return {
            "summary": f"❌ git clone failed:\n{clone_result.stderr}",
            "exit_code": clone_result.returncode,
        }

    # 2. Install the archipel package
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/archipel/archipel"],
        capture_output=True, text=True, timeout=120
    )
    if install.returncode != 0:
        return {
            "summary": f"❌ pip install failed:\n{install.stderr[-2000:]}",
            "exit_code": install.returncode,
        }

    # 3. Change to repo root
    cwd = "/archipel"

    # 4. Compile check on all source modules
    source_files = [
        "archipel/src/archipel/__init__.py",
        "archipel/src/archipel/islands/base_island.py",
        "archipel/src/archipel/islands/lifecycle.py",
        "archipel/src/archipel/islands/specialization.py",
        "archipel/src/archipel/ocean/ocean.py",
        "archipel/src/archipel/current/router.py",
        "archipel/src/archipel/current/courant.py",
        "archipel/src/archipel/current/topk_curriculum.py",
        "archipel/src/archipel/training/loop.py",
        "archipel/src/archipel/training/loop_lifecycle.py",
        "archipel/src/archipel/training/csv_logger.py",
        "archipel/src/archipel/utils/specialization_matrix.py",
        "archipel/train.py",
    ]

    print("\n📦 Compile check…")
    errors = 0
    for src in source_files:
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", f"{cwd}/{src}"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            print(f"❌ {src}: FAILED\n{r.stderr[:300]}")
            errors += 1
        else:
            print(f"✅ {src}")
    if errors > 0:
        print(f"\n⚠ {errors} compile error(s) found — continuing to tests anyway")

    # 5. Run the test suites
    test_files = ["test_topk_curriculum.py"]
    if not quick:
        test_files += ["test_specialization.py", "test_train_script.py"]

    all_output = []
    all_passed = True
    total_pass = 0
    total_fail = 0

    for tf in test_files:
        print(f"\n🔬 Running {tf}…")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", tf, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=300,
            cwd=cwd,
        )
        # Show output (last 30 lines)
        out_lines = r.stdout.strip().split("\n")
        for line in out_lines[-25:]:
            print(f"  {line}")
        if r.stderr:
            print(f"  STDERR: {r.stderr[:300]}")

        # Parse summary line
        summary_line = [l for l in out_lines if "passed" in l or "failed" in l]
        if summary_line:
            all_output.append(f"{tf}: {summary_line[-1]}")
        else:
            all_output.append(f"{tf}: exit_code={r.returncode}")

        if r.returncode == 0:
            total_pass += 1
        else:
            total_fail += 1
            all_passed = False

    # 6. Summary
    lines = [f"📊 RESULTS ({len(test_files)} test suites)"]
    for l in all_output:
        lines.append(f"   {l}")
    lines.append(f"   ─────────────────")
    lines.append(f"   Suites passed: {total_pass}/{len(test_files)}")
    lines.append(f"   Suites failed: {total_fail}/{len(test_files)}")

    return {
        "summary": "\n".join(lines),
        "exit_code": 0 if all_passed else 1,
    }
