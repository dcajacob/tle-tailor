#!/usr/bin/env python3
"""Generate script-backed companions without changing preserved notebooks.

Run from a source checkout. ``--check`` reports missing or stale companions
without writing files. The manifest inventories every *_script.py entrypoint;
all other original notebooks remain explicitly listed as preserved studies.
"""

import argparse
import ast
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


EXAMPLES = Path("tle_tailor/examples")
MANIFEST = EXAMPLES / "workflows.json"
LOCAL_OVERLAY = Path(".local/workflows.private.json")
SCRIPTS = EXAMPLES / "scripts"
NOTEBOOKS = EXAMPLES / "notebooks"
MAINTAINED = NOTEBOOKS / "maintained"
STATUSES = {"maintained", "diagnostic", "historical-requires-legacy-api"}


def matches(path, pattern):
    """Match policy globs by component, with ** spanning zero or more parts."""
    parts, patterns = path.split("/"), pattern.split("/")
    def walk(i, j):
        if j == len(patterns):
            return i == len(parts)
        if patterns[j] == "**":
            return walk(i, j + 1) or (i < len(parts) and walk(i + 1, j))
        return i < len(parts) and fnmatch.fnmatchcase(parts[i], patterns[j]) and walk(i + 1, j + 1)
    return walk(0, 0)


def private_patterns(root):
    """Public policy identifies local-only paths without reading private inputs."""
    policy = root / "distribution-policy.json"
    patterns = json.loads(policy.read_text()).get("private_local_globs", []) if policy.is_file() else []
    if not isinstance(patterns, list) or any(
        not isinstance(pattern, str) or not pattern or PurePosixPath(pattern).is_absolute()
        or ".." in PurePosixPath(pattern).parts or "\\" in pattern for pattern in patterns
    ):
        raise ValueError("Private exclusions must be relative project globs")
    return patterns


def is_private(path, patterns):
    return any(matches(path, pattern) for pattern in patterns)


def _inventory(root, *, distributed):
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    patterns = private_patterns(root)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported workflow manifest schema")
    for entry in manifest.get("workflows", []):
        if any(is_private(entry[key], patterns) for key in ("script", "original_notebook", "companion")):
            raise ValueError("Private path in public workflow manifest")
    if any(is_private(entry["notebook"], patterns) for entry in manifest.get("preserved_studies", [])):
        raise ValueError("Private study in public workflow manifest")
    for entry in manifest.get("standalone_companions", []):
        if (is_private(entry["notebook"], patterns)
                or is_private(entry["module"].replace(".", "/") + ".py", patterns)):
            raise ValueError("Private standalone companion in public workflow manifest")
    # Distributed checks never read or validate the caller's local overlay.
    if not distributed:
        overlay_path = root / LOCAL_OVERLAY
        if overlay_path.exists() or overlay_path.is_symlink():
            if overlay_path.is_symlink() or not overlay_path.is_file():
                raise ValueError("Private workflow overlay must be a regular local file")
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            if overlay.get("schema_version") != 1:
                raise ValueError("Unsupported private workflow overlay schema")
            for entry in overlay.get("workflows", []):
                if not all(is_private(entry[key], patterns) for key in ("script", "original_notebook", "companion")):
                    raise ValueError("Private workflow paths require explicit distribution exclusions")
            if any(not is_private(entry["notebook"], patterns) for entry in overlay.get("preserved_studies", [])):
                raise ValueError("Private study requires an explicit distribution exclusion")
            for key in ("workflows", "preserved_studies"):
                manifest[key] = manifest.get(key, []) + overlay.get(key, [])
    return manifest, patterns


def _path(root, value, *, within=None):
    """Accept only source-relative paths inside the intended directory."""
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"Expected a relative project path: {value}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the source checkout: {value}")
    if within is not None and not path.resolve().is_relative_to((root / within).resolve()):
        raise ValueError(f"Path must be within {within}: {value}")
    return path


def load_manifest(root, *, distributed=False, require_originals=None):
    root = Path(root)
    manifest, patterns = _inventory(root, distributed=distributed)
    if require_originals is None:
        require_originals = not distributed
    entries = manifest.get("workflows", [])
    seen = {key: set() for key in ("id", "script", "companion")}
    originals = set()
    for entry in entries:
        for key in seen:
            if entry[key] in seen[key]:
                raise ValueError(f"Duplicate {key}: {entry[key]}")
            seen[key].add(entry[key])
        if entry["status"] not in STATUSES:
            raise ValueError(f"Unknown workflow status: {entry['status']}")
        script = _path(root, entry["script"], within=SCRIPTS)
        original = _path(root, entry["original_notebook"], within=NOTEBOOKS)
        companion = _path(root, entry["companion"], within=MAINTAINED)
        if not script.is_file() or not script.name.endswith("_script.py"):
            raise ValueError(f"Missing script entrypoint: {entry['script']}")
        if original.suffix != ".ipynb" or (require_originals and not original.is_file()):
            raise ValueError(f"Missing preserved notebook: {entry['original_notebook']}")
        if companion.suffix != ".ipynb":
            raise ValueError(f"Companion must be a notebook: {entry['companion']}")
        module = ".".join(PurePosixPath(entry["script"]).with_suffix("").parts)
        if entry["module"] != module:
            raise ValueError(f"Module does not match script path: {entry['id']}")
        if not isinstance(entry["arguments"], list) or not all(isinstance(arg, str) for arg in entry["arguments"]):
            raise ValueError(f"Arguments must be strings: {entry['id']}")
        originals.add(entry["original_notebook"])
    actual_scripts = {p.relative_to(root).as_posix() for p in (root / SCRIPTS).rglob("*_script.py")
                      if not distributed or not is_private(p.relative_to(root).as_posix(), patterns)}
    if seen["script"] != actual_scripts:
        missing = sorted(actual_scripts - seen["script"])
        obsolete = sorted(seen["script"] - actual_scripts)
        raise ValueError(f"Workflow script inventory differs: missing={missing}, obsolete={obsolete}")
    preserved = manifest.get("preserved_studies", [])
    for entry in preserved:
        path = _path(root, entry["notebook"], within=NOTEBOOKS)
        if path.suffix != ".ipynb" or (require_originals and not path.is_file()):
            raise ValueError(f"Missing preserved study: {entry['notebook']}")
        if entry["notebook"] in originals:
            raise ValueError(f"Notebook appears in both inventories: {entry['notebook']}")
        originals.add(entry["notebook"])
    actual_originals = {
        p.relative_to(root).as_posix() for p in (root / NOTEBOOKS).rglob("*.ipynb")
        if not p.is_relative_to(root / MAINTAINED) and ".ipynb_checkpoints" not in p.parts
    }
    if require_originals and originals != actual_originals:
        standalone = standalone_notebooks(root, manifest)
        if originals != actual_originals - standalone:
            raise ValueError("Preserved notebook inventory differs from source checkout")
    else:
        standalone = standalone_notebooks(root, manifest)
    if originals & standalone:
        raise ValueError("Standalone companion appears in the preserved notebook inventory")
    return manifest


def standalone_notebooks(root, manifest):
    """Validate manually authored dispatch companions outside generated history."""
    result = set()
    for entry in manifest.get("standalone_companions", []):
        name, module = entry["notebook"], entry["module"]
        path = _path(root, name, within=NOTEBOOKS)
        if name in result or path.suffix != ".ipynb" or not path.is_file() or path.is_relative_to(root / MAINTAINED):
            raise ValueError(f"Invalid standalone companion: {name}")
        if not module.startswith("tle_tailor.examples.scripts.") or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError(f"Invalid standalone module: {module}")
        script = _path(root, module.replace(".", "/") + ".py", within=SCRIPTS)
        if not script.is_file():
            raise ValueError(f"Missing standalone module: {module}")
        notebook = json.loads(path.read_text())
        code = "\n".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                         for cell in notebook["cells"] if cell["cell_type"] == "code")
        body = ast.parse(code).body
        # Keep the editable argument list, while excluding copied inline solvers.
        if not (len(body) == 3 and isinstance(body[0], ast.ImportFrom)
                and body[0].module == module and len(body[0].names) == 1
                and body[0].names[0].name == "main" and body[0].names[0].asname is None
                and isinstance(body[1], ast.Assign) and len(body[1].targets) == 1
                and isinstance(body[1].targets[0], ast.Name) and body[1].targets[0].id == "arguments"
                and isinstance(body[2], ast.Expr) and isinstance(body[2].value, ast.Call)
                and isinstance(body[2].value.func, ast.Name) and body[2].value.func.id == "main"
                and len(body[2].value.args) == 1 and isinstance(body[2].value.args[0], ast.Name)
                and body[2].value.args[0].id == "arguments" and not body[2].value.keywords):
            raise ValueError(f"Standalone companion must dispatch to its declared main: {name}")
        arguments = ast.literal_eval(body[1].value)
        if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
            raise ValueError(f"Standalone arguments must be strings: {name}")
        result.add(name)
    return result


def render_notebook(entry, root):
    """Keep all numerical behavior in the maintained module, with no saved output."""
    status = {
        "maintained": "Maintained script workflow",
        "diagnostic": "GPS fixed-BSTAR workflow — inspect physical fit errors alongside legacy convergence diagnostics",
        "historical-requires-legacy-api": "Historical workflow — separate legacy GMAT API required",
    }[entry["status"]]
    requirements = "; ".join(entry["requirements"])
    differences = "\n".join(f"- {note}" for note in entry["original_differences"])
    introduction = (
        f"# {entry['title']}\n\n**{status}.**\n\n"
        f"This generated companion runs `{entry['module']}`. Its fitting and propagation "
        "code is shared with the command-line workflow; there is no copied solver in this notebook.\n\n"
        f"Requirements: {requirements}.\n\n"
        f"Input category: **{entry['input_category']}**. {entry['input_notes']}\n\n"
        f"The original `{entry['original_notebook']}` is preserved separately. "
        "This companion covers the named script workflow, not every original research cell.\n\n"
        f"Differences and retained studies:\n{differences}\n\n"
        "Run with an environment where TLE Tailor is installed. Generated figures and data use "
        "`TLE_TAILOR_OUTPUT_DIR` (or `./output`). Where supplied, `--no-plots` saves figures "
        "without opening windows; remove it from `arguments` to display them. Scripts that "
        "offer live Space-Track comparisons use configured credentials if available. "
        "A legacy GMAT API workflow needs its own compatible kernel; do not mix its native bindings "
        "with standalone gmatpy in one kernel."
    )
    invocation = (
        "import runpy\nimport sys\n\n"
        f"workflow_module = {entry['module']!r}\n"
        f"arguments = {entry['arguments']!r}\n\n"
        "# Restore Jupyter's arguments even if the workflow rejects its inputs.\n"
        "previous_argv = sys.argv[:]\n"
        "try:\n"
        "    sys.argv[:] = [workflow_module, *arguments]\n"
        "    try:\n"
        "        runpy.run_module(workflow_module, run_name='__main__', alter_sys=True)\n"
        "    except SystemExit as exit_status:\n"
        "        if exit_status.code not in (None, 0):\n"
        "            raise\n"
        "finally:\n"
        "    sys.argv[:] = previous_argv\n"
    )
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "workflow-guide", "metadata": {}, "source": introduction.splitlines(keepends=True)},
            {"cell_type": "code", "id": "run-workflow", "metadata": {}, "execution_count": None,
             "outputs": [], "source": invocation.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (TLE Tailor)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "tle_tailor_workflow": {
                "schema_version": 1, "id": entry["id"], "script": entry["script"],
                "script_sha256": hashlib.sha256((root / entry["script"]).read_bytes()).hexdigest(),
                "status": entry["status"], "input_category": entry["input_category"],
                "generated_by": "scripts/sync_workflow_notebooks.py",
            },
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def synchronize(root, *, check=False, distributed=False):
    manifest = load_manifest(root, distributed=distributed)
    patterns = private_patterns(root)
    expected = {root / entry["companion"]: render_notebook(entry, root) for entry in manifest["workflows"]}
    actual = {p for p in (root / MAINTAINED).rglob("*.ipynb")
              if ".ipynb_checkpoints" not in p.parts
              and (not distributed or not is_private(p.relative_to(root).as_posix(), patterns))}
    # Never silently remove notebooks, including a user's added companion.
    unexpected = sorted(actual - set(expected))
    if unexpected:
        raise ValueError("Unlisted maintained notebooks: " + ", ".join(str(p.relative_to(root)) for p in unexpected))
    changed = []
    for path, contents in expected.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != contents:
            changed.append(path.relative_to(root).as_posix())
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report drift without writing any file")
    parser.add_argument("--distributed", action="store_true", help="check scripts and companions without requiring preserved source-history notebooks")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="source checkout root")
    args = parser.parse_args(argv)
    if args.distributed and not args.check:
        parser.error("--distributed requires --check")
    try:
        changed = synchronize(args.root.resolve(), check=args.check, distributed=args.distributed)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"Workflow notebook sync failed: {exc}", file=sys.stderr)
        return 1
    if changed:
        action = "Stale or missing" if args.check else "Generated"
        print(f"{action} workflow companions ({len(changed)}):\n" + "\n".join(changed))
    else:
        print("Workflow companions are synchronized.")
    if args.distributed:
        print("Distributed script/companion check only; preserved source studies were not checked or executed.")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
