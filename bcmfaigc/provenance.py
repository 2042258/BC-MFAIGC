"""Record current executed source files without claiming historical byte equality."""

from hashlib import sha256
import json
from pathlib import Path


def build_source_manifest(experiment_root):
    """Inventory experiment adapters and the complete shared Python package.

    This is source identification, not numerical equivalence, dataset integrity
    or historical-paper reproduction verification. Dependency versions and
    effective arguments require their own records.
    """
    experiment_root = Path(experiment_root)
    package_root = Path(__file__).resolve().parent
    roots = {
        "experiment/BFL": experiment_root / "BFL",
        "experiment/models": experiment_root / "models",
        "shared": package_root,
    }
    files = {}
    for label, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing source directory: {label}")
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.relative_to(root).parts:
                continue
            if path.is_symlink():
                raise ValueError(
                    f"Source symlink requires explicit provenance: {label}"
                )
            name = f"{label}/{path.relative_to(root).as_posix()}"
            files[name] = sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "historical_reference_status": "not_evaluated",
        "coverage": "fresh_run_startup_only",
        "files": files,
    }


def write_source_manifest(output_path, experiment_root):
    """Write an additive inventory; never overwrite a historical fingerprint."""
    output_path = Path(output_path)
    if output_path.name == "code_fingerprint.json":
        raise ValueError("The current inventory must not replace code_fingerprint.json")
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite an existing source inventory")
    document = build_source_manifest(experiment_root)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document
