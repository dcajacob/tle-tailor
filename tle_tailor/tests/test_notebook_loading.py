"""Check the Jupyter opening path that fresh kernel execution does not cover."""

import os
import json

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("jupyter_server")

from scripts.validate_notebooks import check_jupyter_load


def test_notebook_load_creates_checkpoint_without_changing_source(tmp_path):
    path = tmp_path / "example.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")]), path)
    original = path.read_bytes()
    result = check_jupyter_load(path, root=tmp_path)
    assert result["read"] and result["checkpoint_created"]
    assert path.read_bytes() == original
    assert (tmp_path / ".ipynb_checkpoints/example-checkpoint.ipynb").read_bytes() == original


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="requires ordinary POSIX filesystem permissions")
def test_notebook_load_reports_unwritable_checkpoint_directory(tmp_path):
    path = tmp_path / "example.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), path)
    checkpoints = tmp_path / ".ipynb_checkpoints"
    checkpoints.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError):
            check_jupyter_load(path, root=tmp_path)
    finally:
        checkpoints.chmod(0o700)


def test_default_validation_uses_only_self_contained_script_companions():
    from scripts.validate_notebooks import maintained_notebooks, NOTEBOOKS
    paths = maintained_notebooks()
    assert len(paths) == 8
    assert paths[0] == NOTEBOOKS / 'deployment.ipynb'
    assert all(path.is_relative_to(NOTEBOOKS / 'maintained') for path in paths[1:])
    assert all(path.parent.name in {'basic_examples', 'gmat_integration'} for path in paths[1:])
    from scripts import validate_notebooks
    public = json.loads((NOTEBOOKS.parent / 'workflows.json').read_text())['workflows']
    local_overlay = validate_notebooks.ROOT / '.local/workflows.private.json'
    private = json.loads(local_overlay.read_text())['workflows'] if local_overlay.is_file() else []
    expected = 1 + sum(item['status'] == 'maintained' for item in public + private)
    assert len(maintained_notebooks(include_local_data=True)) == expected
