"""Build a static public site from an explicit page and download inventory.

This hook reads source text and JSON only. It never imports application code,
executes notebooks, opens local overlays, or discovers files outside docs/.
"""
from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File, Files


PUBLIC_PAGES = (
    "accuracy.md", "deployment.md", "ephemerides.md", "gmatpy.md",
    "installation.md", "local_gmat.md", "propagators.md", "redistribution.md",
    "release-notes.md", "spacetrack.md", "support.md", "theory.md", "workflows.md",
    "walkthrough-deployment.md", "walkthrough-gps.md", "walkthrough-edie.md",
    "validation-results.md", "performance.md",
)
SOURCE_ASSETS = (
    "tle_tailor/examples/scripts/deployment.py",
    "tle_tailor/examples/scripts/_fitting.py",
    "tle_tailor/examples/scripts/gmat_integration/gmatpy_state_to_tle.py",
    "LICENSE", "NOTICE", "licenses/python-sgp4-MIT.txt",
)
OPTIONAL_DRIVERS = (
    "docs/examples/deployment_to_tle.py", "docs/examples/gps_to_tle.py",
    "docs/examples/novatel_edie_to_tle.py", "docs/examples/requirements-edie.txt",
)
OPTIONAL_SITE_ASSETS = ("docs/assets/validation-results.json", "docs/assets/fitting-performance.json")
PUBLIC_NOTEBOOKS = (
    "tle_tailor/examples/notebooks/deployment.ipynb",
    "tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_coe_fd.ipynb",
    "tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_coe_jax.ipynb",
    "tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_eqn_fd.ipynb",
    "tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_eqn_jax.ipynb",
    "tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_eqn_no_bstar.ipynb",
    "tle_tailor/examples/notebooks/maintained/gmat_integration/gmat_opm_prop_fd_example.ipynb",
    "tle_tailor/examples/notebooks/maintained/gmat_integration/gmat_opm_prop_jax_example.ipynb",
    "tle_tailor/examples/notebooks/maintained/gmat_integration/tle_from_gmat.ipynb",
    "tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_fd.ipynb",
    "tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_jax_match.ipynb",
    "tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_jax.ipynb",
    "tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_no_bstar.ipynb",
    "tle_tailor/examples/notebooks/maintained/icesat_examples/tlefit_eqn_gps_fd_icesat.ipynb",
    "tle_tailor/examples/notebooks/maintained/icesat_examples/tlefit_eqn_gps_jax_icesat.ipynb",
)
PREVIEW_BANNER = (
    "> **Release candidate:** Install [0.1.0rc1 from PyPI]"
    "(https://pypi.org/project/tle-tailor/0.1.0rc1/) with the explicit "
    "`tle-tailor==0.1.0rc1` pin for these commands. "
    "The supported target is standard CPython 3.14 on Linux x86_64. "
    "Public `main` remains the historical source; use the prerelease or its matching tag.\n\n"
)
_ORIGINS: dict[str, str] = {}
_DESTINATIONS: dict[str, str] = {}
_INLINE_LINK = re.compile(r"(?P<before>!?\[[^\]\n]*\]\()(?P<target>[^)\n]+)(?P<after>\))")
_REFERENCE_LINK = re.compile(r"^(?P<before> {0,3}\[[^\]\n]+\]:\s*)(?P<target><[^>]+>|\S+)", re.MULTILINE)


def _read_public(root: Path, name: str) -> bytes:
    """Read one declared regular file without allowing symlink/path escapes."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise PluginError(f"Unsafe documentation source path: {name}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise PluginError(f"Documentation source must not be a symlink: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise PluginError(f"Missing regular public documentation source: {name}")
    return path.read_bytes()


def _notebook_assets(root: Path) -> tuple[str, ...]:
    manifest = json.loads(_read_public(root, "tle_tailor/examples/workflows.json"))
    workflows = manifest.get("workflows", [])
    standalone = manifest.get("standalone_companions", [])
    if len(workflows) != 14 or len(standalone) != 1:
        raise PluginError("Public documentation requires exactly 14 workflows and one standalone notebook")
    paths = [item.get("companion") for item in workflows] + [item.get("notebook") for item in standalone]
    if len(set(paths)) != 15 or set(paths) != set(PUBLIC_NOTEBOOKS):
        raise PluginError("Public notebook manifest differs from the explicit documentation allowlist")
    for name in PUBLIC_NOTEBOOKS:
        notebook = json.loads(_read_public(root, name))
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise PluginError(f"Invalid public notebook: {name}")
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                raise PluginError(f"Public notebook contains saved output or execution count: {name}")
    return PUBLIC_NOTEBOOKS


def on_files(files: Files, config, **kwargs) -> Files:
    """Replace discovery with the exact public page/download inventory."""
    root = Path(config.config_file_path).resolve().parent
    if Path(config.docs_dir).resolve() != root / "docs":
        raise PluginError("Public documentation must use the dedicated docs/ source directory")
    selected = Files([])
    _ORIGINS.clear()
    _DESTINATIONS.clear()
    for page in PUBLIC_PAGES:
        _read_public(root, f"docs/{page}")
        source = files.get_file_from_path(page)
        if source is None:
            raise PluginError(f"Required public page is not discoverable: {page}")
        selected.append(source)
        _ORIGINS[page] = f"docs/{page}"
        _DESTINATIONS[f"docs/{page}"] = page
    for target, source in (("index.md", "README.md"), ("quickstart.md", "QUICKSTART.md")):
        selected.append(File.generated(config, target, content=_read_public(root, source).decode("utf-8")))
        _ORIGINS[target] = source
        _DESTINATIONS[source] = target
        _DESTINATIONS[f"docs/{target}"] = target
    assets = list(SOURCE_ASSETS) + list(_notebook_assets(root))
    assets += [name for name in OPTIONAL_DRIVERS if (root / name).exists()]
    for source in assets:
        destination = f"assets/source/{source}"
        selected.append(File.generated(config, destination, content=_read_public(root, source)))
        _DESTINATIONS[source] = destination
    for source in OPTIONAL_SITE_ASSETS:
        if (root / source).exists():
            content = _read_public(root, source)
            json.loads(content)  # This public receipt is JSON, never executable content.
            destination = source.removeprefix("docs/")
            selected.append(File.generated(config, destination, content=content))
            _DESTINATIONS[source] = destination
    # MkDocs adds theme/search resources before this event. Re-add those from
    # the installed theme after filtering repository files so CSS and JS remain
    # available without allowing arbitrary docs/ files into the site.
    selected.add_files_from_theme(config.theme.get_env(), config)
    return selected


def _rewrite_target(target: str, origin: str, page_uri: str) -> str:
    # Preserve optional Markdown link titles and angle-wrapped destinations.
    match = re.fullmatch(r'(?P<url><[^>]+>|\S+)(?P<title>\s+["\'][^\n]*["\'])?', target.strip())
    if match is None:
        raise PluginError(f"Unrecognized documentation link in {origin}: {target}")
    url = match["url"]
    wrapped = url.startswith("<")
    parsed = urlsplit(url[1:-1] if wrapped else url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return target
    if parsed.path.startswith("/"):
        raise PluginError(f"Documentation link must be relative or an explicit public URL: {origin}: {url}")
    source = posixpath.normpath(posixpath.join(posixpath.dirname(origin), unquote(parsed.path)))
    destination = _DESTINATIONS.get(source)
    if destination is None:
        raise PluginError(f"Link targets an undeclared public page/download: {origin}: {url}")
    relative = posixpath.relpath(destination, posixpath.dirname(page_uri) or ".")
    rewritten = urlunsplit(("", "", relative, parsed.query, parsed.fragment))
    return (f"<{rewritten}>" if wrapped else rewritten) + (match["title"] or "")


def on_page_markdown(markdown: str, page, config, **kwargs) -> str:
    """Rebase repository-relative links without rewriting code examples."""
    uri = page.file.src_uri
    origin = _ORIGINS[uri]

    def replace(match):
        return match["before"] + _rewrite_target(match["target"], origin, uri) + match.groupdict().get("after", "")

    blocks = re.split(r"(^ {0,3}(?:`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?:`{3,}|~{3,})\s*$)", markdown, flags=re.MULTILINE | re.DOTALL)
    for index in range(0, len(blocks), 2):
        # A link label may itself contain inline code. Protect only link syntax
        # entirely inside a code span, not a surrounding Markdown link.
        code = [(match.start(), match.end()) for match in re.finditer(r"(`+)[^`\n]*\1", blocks[index])]

        def inline_replace(match):
            if any(start <= match.start() and match.end() <= end for start, end in code):
                return match[0]
            return replace(match)

        blocks[index] = _REFERENCE_LINK.sub(replace, _INLINE_LINK.sub(inline_replace, blocks[index]))
    return PREVIEW_BANNER + "".join(blocks)
