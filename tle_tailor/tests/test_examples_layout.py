"""Static example layout and scratch-path contracts; no workflow execution."""

import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "tle_tailor" / "examples"
NOTEBOOKS = EXAMPLES / "notebooks"
SCRIPTS = EXAMPLES / "scripts"


def authored_notebooks():
    return sorted(path for path in NOTEBOOKS.rglob("*.ipynb")
                  if ".ipynb_checkpoints" not in path.parts)


def historical_notebooks():
    return [path for path in authored_notebooks()
            if "maintained" not in path.relative_to(NOTEBOOKS).parts]


def source_text(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def relocated_scratch_notebooks():
    return [path for path in historical_notebooks()
            if any("def _example_sat_file(" in source_text(cell)
                   for cell in json.loads(path.read_text())["cells"])]


def test_notebooks_and_scripts_have_separate_roots():
    assert NOTEBOOKS.is_dir()
    assert SCRIPTS.is_dir()
    assert authored_notebooks()
    assert any(SCRIPTS.rglob("*.py"))
    assert not [path for path in NOTEBOOKS.rglob("*.py")
                if ".ipynb_checkpoints" not in path.parts]
    assert not list(SCRIPTS.rglob("*.ipynb"))
    assert not list(EXAMPLES.rglob("*.bsp")), "Use the single shared source-only kernel"


def test_notebooks_remain_structurally_valid_json():
    for path in authored_notebooks():
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4, path
        assert isinstance(notebook["metadata"], dict), path
        assert isinstance(notebook["cells"], list), path
        for cell in notebook["cells"]:
            assert cell["cell_type"] in {"code", "markdown", "raw"}, path
            assert isinstance(cell["metadata"], dict), path
            assert isinstance(cell["source"], (str, list)), path
            if isinstance(cell["source"], list):
                assert all(isinstance(line, str) for line in cell["source"]), path
            if cell["cell_type"] == "code":
                assert isinstance(cell["outputs"], list), path


def test_historical_source_references_follow_relocated_layout():
    for path in historical_notebooks():
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            text = source_text(cell)
            assert "tle_tailor.notebooks." not in text, path
            assert "tle_tailor/notebooks" not in text, path
            assert "/home/dan/GIT/tle-tailor/tle_tailor/" not in text, path
            if cell["cell_type"] == "markdown":
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    if "/docs/" in target and not re.match(r"https?://", target):
                        assert (path.parent / target).resolve().is_file(), (path, target)


@pytest.mark.parametrize("path", relocated_scratch_notebooks(),
                         ids=lambda path: path.stem)
def test_scratch_resolver_keeps_generated_writes_out_of_preserved_inputs(path, tmp_path):
    """Exercise only the isolated path helper; never import/run notebook code."""
    notebook = json.loads(path.read_text())
    helper = None
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "def _example_sat_file(" in source_text(cell):
            helper = next(node for node in ast.parse(source_text(cell)).body
                          if isinstance(node, ast.FunctionDef)
                          and node.name == "_example_sat_file")
    assert helper is not None
    output = tmp_path / "output"
    output.mkdir()
    data = tmp_path / "data"
    preserved = data / "examples" / "gmat_integration" / "sat0.fth"
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(b"preserved input")
    namespace = {"_example_config": SimpleNamespace(output_path=output, data_path=data)}
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    exec(compile(module, "<isolated example path helper>", "exec"), namespace)
    resolve = namespace["_example_sat_file"]
    assert resolve(0) == preserved
    generated = resolve(0, generated=True)
    assert generated == output / "sat0.fth"
    generated.write_bytes(b"generated output")
    assert resolve(0) == generated
    assert preserved.read_bytes() == b"preserved input"
    generated.unlink()
    assert resolve(0) == preserved


@pytest.mark.parametrize("path", relocated_scratch_notebooks(),
                         ids=lambda path: path.stem)
def test_scratch_file_calls_share_the_explicit_path_resolver(path):
    notebook = json.loads(path.read_text())
    writes = reads = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse(source_text(cell))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"to_feather", "read_feather"}:
                continue
            assert isinstance(node.args[0], ast.Call), path
            call = node.args[0]
            assert isinstance(call.func, ast.Name) and call.func.id == "_example_sat_file", path
            if node.func.attr == "to_feather":
                writes += 1
                assert any(k.arg == "generated" and isinstance(k.value, ast.Constant)
                           and k.value.value is True for k in call.keywords), path
            else:
                reads += 1
                assert not call.keywords, path
    assert writes and reads, path
