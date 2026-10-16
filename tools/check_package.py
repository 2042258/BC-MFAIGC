"""Check this package without importing or running its experiment code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL_PATTERN = re.compile(r"(?:https?|ftp|ssh|git)" + r"://|www\.", re.I)
EMAIL_PATTERN = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PRIVATE_PATH_PATTERN = re.compile(
    r"/(?:home|Users|root|mnt/[a-z]|content|workspace|opt)/"
    r"|(?<![\w])[A-Za-z]:[\\/]|~/(?:work|projects?)/",
    re.I,
)
PLACEHOLDER_PATTERN = re.compile(
    r"(?:URL_REQUIRED_\d+|PATH_REQUIRED_\d+|REMOTE_HOST_REQUIRED)"
)


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


def check() -> dict:
    issues = []
    counts = {"files": 0, "python": 0, "json": 0, "shell": 0}
    placeholders = []
    bash = shutil.which("bash")
    # Regression fixture: a log parser expression is not a Windows path.
    if PRIVATE_PATH_PATTERN.search(r"value:\s+"):
        issues.append("Path scanner misclassifies a log-parsing expression")

    paths = sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(ROOT).parts
    )
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            issues.append(f"Symlink is not allowed in the review payload: {relative}")
            continue
        counts["files"] += 1
        try:
            text = read_text(path)
        except UnicodeDecodeError:
            issues.append(f"Unexpected binary file: {relative}")
            continue
        if path.suffix == ".py":
            counts["python"] += 1
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as error:
                issues.append(f"Python syntax: {relative}:{error.lineno}")
        elif path.suffix == ".json":
            counts["json"] += 1
            try:
                json.loads(text)
            except ValueError:
                issues.append(f"JSON syntax: {relative}")
        elif path.suffix in {".sh", ".bash"}:
            counts["shell"] += 1
            if bash:
                result = subprocess.run(
                    [bash, "-n", str(path)], capture_output=True, text=True
                )
                if result.returncode:
                    issues.append(f"Shell syntax: {relative}")
            else:
                issues.append(f"Shell syntax not checked; bash missing: {relative}")
        if relative.startswith(
            ("experiments/", "analysis/", "bcmfaigc/", "configs/", "supplemental/")
        ):
            for name, pattern in [
                ("URL", URL_PATTERN),
                ("email", EMAIL_PATTERN),
                ("user-specific path", PRIVATE_PATH_PATTERN),
            ]:
                if pattern.search(text):
                    issues.append(f"Residual {name}: {relative}")
            matches = PLACEHOLDER_PATTERN.findall(text)
            if matches:
                placeholders.append({"file": relative, "occurrences": len(matches)})

    for label in ("comparison", "byzantine_ratio", "krum", "trust_weighting"):
        for dependency in (
            "BFL/main.py",
            "BFL/Device.py",
            "BFL/DatasetLoad.py",
            "models/Models.py",
        ):
            if not (ROOT / "experiments" / label / dependency).is_file():
                issues.append(
                    f"Missing training source dependency: {label}/{dependency}"
                )
    for dependency in (
        "Block.py",
        "Blockchain.py",
        "Device.py",
        "main.py",
        "stable_hash.py",
    ):
        if not (
            ROOT / "experiments/protocol_control/source_snapshot" / dependency
        ).is_file():
            issues.append(f"Missing protocol source snapshot: {dependency}")

    index = json.loads(read_text(ROOT / "experiment_index.json"))
    for item in index:
        root = ROOT / item["code_root"]
        for key in ("runner", "manifest"):
            base = ROOT if key == "manifest" else root
            if not (base / item[key]).is_file():
                issues.append(f"Missing {key}: {item['workflow']}")
        manifest_path = ROOT / item["manifest"]
        if manifest_path.is_file():
            manifest = json.loads(read_text(manifest_path))
            available = {row["id"] for row in manifest.get("runs", [])}
            selected = item["run_ids"]
            if len(selected) != len(set(selected)) or not set(selected) <= available:
                issues.append(f"Invalid run selection: {item['workflow']}")
            if item["run_count"] != len(selected):
                issues.append(f"Wrong run count: {item['workflow']}")

    checksum_file = ROOT / "MANIFEST.sha256"
    expected = {}
    if not checksum_file.exists():
        issues.append("Missing MANIFEST.sha256")
    else:
        for line in read_text(checksum_file).splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
            path = ROOT / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                issues.append(f"Checksum mismatch: {name}")
        actual = {
            p.relative_to(ROOT).as_posix()
            for p in paths
            if p != checksum_file and "__pycache__" not in p.parts
        }
        if actual != set(expected):
            issues.append("Packaged file set differs from MANIFEST.sha256")

    metadata = json.loads(read_text(ROOT / "package_metadata.json"))
    blockers = metadata.get("release_blockers", [])
    return {
        "static_checks_passed": not issues,
        "counts": counts,
        "issues": issues,
        "files_with_functional_placeholders": placeholders,
        "release_status": metadata["status"],
        "release_ready": not issues and not placeholders and not blockers,
        "release_blockers": blockers,
        "checker_executed_experiments": False,
        "verification": metadata.get("verification", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Also fail when public-release prerequisites remain unresolved",
    )
    args = parser.parse_args()
    report = check()
    print(json.dumps(report, indent=2))
    if not report["static_checks_passed"]:
        return 1
    if args.strict_release and not report["release_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
