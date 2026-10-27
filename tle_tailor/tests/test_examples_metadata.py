"""Example module locations and mandatory dependency metadata, without fits."""
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "tle_tailor/examples"


def test_primary_dependency_and_compatibility_extra_match_lock():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = metadata["project"]
    assert "gmatpy==0.2.1" in project["dependencies"]
    assert project["optional-dependencies"]["gmat"] == ["gmatpy==0.2.1"]
    assert metadata["tool"]["uv"]["no-build-package"] == ["gmatpy"]
    assert metadata["tool"]["uv"]["default-groups"] == []
    assert project["scripts"]["tle-tailor"] == "tle_tailor.cli:main"
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    package = next(item for item in lock["package"] if item["name"] == "tle-tailor")
    assert any(item["name"] == "gmatpy" for item in package["dependencies"])
    assert any(item["name"] == "gmatpy" and item["specifier"] == "==0.2.1"
               and "marker" not in item for item in package["metadata"]["requires-dist"])


def test_public_example_modules_resolve_from_script_tree_without_execution():
    manifest = json.loads((EXAMPLES / "workflows.json").read_text())
    assert len(manifest["workflows"]) == 14
    before = set(sys.modules)
    for entry in manifest["workflows"]:
        script = ROOT / entry["script"]
        assert script.is_relative_to(EXAMPLES / "scripts")
        assert (ROOT / entry["original_notebook"]).is_relative_to(EXAMPLES / "notebooks")
        assert entry["module"].startswith("tle_tailor.examples.scripts.")
        spec = importlib.util.find_spec(entry["module"])
        assert spec is not None and Path(spec.origin).resolve() == script.resolve()
        assert entry["module"] not in set(sys.modules) - before
    assert not list((EXAMPLES / "notebooks").rglob("*_script.py"))


def test_split_private_script_tree_and_local_assets_remain_excluded():
    policy = json.loads((ROOT / "distribution-policy.json").read_text())
    private = "tle_tailor/examples/scripts/hawk_examples/**"
    assert private in policy["private_local_globs"]
    assert "tle_tailor/data/examples/**" in policy["exclude_globs"]
    backend = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["build-backend"]
    for kind in ("source-exclude", "wheel-exclude"):
        assert private in backend[kind]
        assert "tle_tailor/data/examples/**" in backend[kind]
