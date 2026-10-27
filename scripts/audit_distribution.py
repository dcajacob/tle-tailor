#!/usr/bin/env python3
"""Fail closed on undeclared distribution payloads, data, and missing notices.

Run --source-only before building, then audit both actual wheel and sdist.
This checks the declared distribution boundary, not whole-checkout permission.
"""
import argparse
import ast
import base64
import configparser
import csv
import datetime as dt
from email.parser import BytesParser
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tomllib
import zipfile


def matches(path, pattern):
    """Glob by path component: * cannot cross /, ** can match zero components."""
    parts, pats = path.split('/'), pattern.split('/')
    def walk(i, j):
        if j == len(pats):
            return i == len(parts)
        if pats[j] == '**':
            return walk(i, j + 1) or (i < len(parts) and walk(i + 1, j))
        return i < len(parts) and fnmatch.fnmatchcase(parts[i], pats[j]) and walk(i + 1, j + 1)
    return walk(0, 0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(path):
    return not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts and '\\' not in path


def sanitized_notebook(data):
    notebook = json.loads(data)
    if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
        return False
    return all(cell.get('outputs', []) == [] and cell.get('execution_count') is None for cell in notebook['cells'])


def audit(root, wheel=None, sdist=None):
    policy = json.loads((root/'distribution-policy.json').read_text())
    project = tomllib.loads((root/'pyproject.toml').read_text())
    report = {'schema_version': 1, 'audited_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': policy['scope'], 'checks': [], 'artifacts': {},
              'repository_publication_status': policy['repository_publication_status'],
              'repository_publication_blockers': policy['repository_publication_blockers']}
    def check(name, passed, **details):
        report['checks'].append({'check': name, 'status': 'passed' if passed else 'failed', **details})
    def excluded(name):
        return any(matches(name, p) for p in policy['exclude_globs'] + policy.get('private_local_globs', []))
    def runtime(name):
        return name in policy['runtime_files'] or any(matches(name, p) for p in policy['runtime_globs'])
    def source_allowed(name):
        return runtime(name) or name in policy['source_extra_files'] or any(matches(name, p) for p in policy['source_extra_globs'])
    source_files = {p.relative_to(root).as_posix(): p for p in (root/'tle_tailor').rglob('*') if p.is_file()}
    unknown = sorted(n for n in source_files if not source_allowed(n) and not excluded(n))
    missing = sorted(n for n in policy['runtime_files'] if not (root/n).is_file())
    symlinks = sorted(n for n, p in source_files.items() if p.is_symlink())
    check('all package source files explicitly classified', not unknown and not missing and not symlinks,
          unknown=unknown, missing_declared_runtime=missing, symlinks=symlinks,
          declared_runtime_files=len(policy['runtime_files']), excluded_local_files=sum(excluded(n) for n in source_files))
    absent_extras = sorted(n for n in policy['source_extra_files'] if not (root/n).is_file())
    check('all declared source extras exist before build', not absent_extras, missing=absent_extras)
    notice_errors = [n for n, expected in policy['notice_sha256'].items()
                     if not (root/n).is_file() or sha((root/n).read_bytes()) != expected]
    check('complete project and upstream license notices present with reviewed bytes', not notice_errors,
          notice_sha256=policy['notice_sha256'], mismatches=notice_errors)
    declared_licenses = project['project'].get('license-files', [])
    check('metadata declares every redistribution notice', set(policy['notice_sha256']).issubset(declared_licenses),
          required=sorted(policy['notice_sha256']), declared=declared_licenses)
    vendor_errors = [n for n, expected in policy['vendor_source_sha256'].items()
                     if not (root/n).is_file() or sha((root/n).read_bytes()) != expected]
    check('reviewed borrowed source hashes unchanged', not vendor_errors, mismatches=vendor_errors,
          checked=len(policy['vendor_source_sha256']))
    reference_manifest = root/policy['reference_manifest']
    reference = json.loads(reference_manifest.read_text())
    reference_errors = [n for n, expected in reference['files'].items()
                        if not (reference_manifest.parent/'reference_b9'/n).is_file()
                        or sha((reference_manifest.parent/'reference_b9'/n).read_bytes()) != expected]
    check('frozen public reference files and immutable manifest match reviewed hashes',
          not reference_errors and sha(reference_manifest.read_bytes()) == policy['reference_manifest_sha256'],
          revision=reference['revision'], checked=len(reference['files']), mismatches=reference_errors)
    reference_prefix = (reference_manifest.parent/'reference_b9').relative_to(root).as_posix()+'/'
    invalid_exceptions = [name for name in policy.get('quarantined_import_exceptions', {})
                          if not name.startswith(reference_prefix)
                          or name.removeprefix(reference_prefix) not in reference['files']]
    check('quarantined import exceptions confined to immutable reference source',
          not invalid_exceptions, invalid=invalid_exceptions)
    imports = []
    accepted_reference_imports = []
    notebook_errors = []
    for name, p in source_files.items():
        if not runtime(name) or excluded(name):
            continue
        if name.endswith('.ipynb'):
            try:
                if not sanitized_notebook(p.read_bytes()):
                    notebook_errors.append(name)
            except (ValueError, TypeError):
                notebook_errors.append(name)
        if name.endswith('.py'):
            for node in ast.walk(ast.parse(p.read_bytes(), filename=name)):
                quarantined = ((isinstance(node, ast.ImportFrom) and 'coarse_fit' in (node.module or ''))
                               or (isinstance(node, ast.Import) and any('coarse_fit' in alias.name for alias in node.names)))
                if quarantined:
                    statement = ast.unparse(node)
                    allowed_statements = policy.get('quarantined_import_exceptions', {}).get(name, [])
                    if statement in allowed_statements:
                        accepted_reference_imports.append({'path': name, 'statement': statement})
                    else:
                        imports.append(name)
    check('application code does not import quarantined helper; frozen reference imports declared', not imports,
          files=sorted(set(imports)), frozen_reference_exceptions=accepted_reference_imports)
    check('generated notebook companions contain no outputs or execution counts', not notebook_errors, mismatches=notebook_errors)
    workflow_file = root/'tle_tailor/examples/workflows.json'
    check('canonical workflow manifest is present', workflow_file.is_file(),
          path=workflow_file.relative_to(root).as_posix())
    if workflow_file.is_file():
        workflows = json.loads(workflow_file.read_text())['workflows']
        expected_companions = [item['companion'] for item in workflows]
        expected_scripts = [item['script'] for item in workflows]
        actual_companions = sorted(n for n in source_files if n.startswith('tle_tailor/examples/notebooks/maintained/') and n.endswith('.ipynb') and not excluded(n))
        missing_workflow_files = sorted(n for n in expected_companions+expected_scripts if not safe(n) or excluded(n) or not runtime(n) or not (root/n).is_file())
        invalid_layout = sorted(n for n in expected_scripts if not n.startswith('tle_tailor/examples/scripts/'))
        invalid_layout += sorted(n for n in expected_companions if not n.startswith('tle_tailor/examples/notebooks/maintained/'))
        check('workflow inventory and sanitized companion source set complete',
              not missing_workflow_files and not invalid_layout and len(expected_companions) == len(set(expected_companions))
              and len(expected_scripts) == len(set(expected_scripts))
              and sorted(expected_companions) == actual_companions,
              workflow_count=len(workflows), companion_count=len(actual_companions),
              missing=missing_workflow_files, invalid_layout=invalid_layout,
              undeclared_companions=sorted(set(actual_companions)-set(expected_companions)))
        standalone = json.loads(workflow_file.read_text()).get('standalone_companions', [])
        standalone_notebooks = [item['notebook'] for item in standalone]
        standalone_files = standalone_notebooks + [item['module'].replace('.', '/')+'.py' for item in standalone]
        standalone_errors = sorted(n for n in standalone_files
                                   if not safe(n) or excluded(n) or not runtime(n) or not (root/n).is_file())
        declared_standalone = {n for n in policy['runtime_files']
                               if n.startswith('tle_tailor/examples/notebooks/') and n.endswith('.ipynb')
                               and not n.startswith('tle_tailor/examples/notebooks/maintained/')}
        check('standalone script-backed companions are explicitly inventoried',
              not standalone_errors and len(standalone_notebooks) == len(set(standalone_notebooks))
              and set(standalone_notebooks) == declared_standalone,
              notebook_count=len(standalone_notebooks), missing=standalone_errors,
              unlisted_runtime_notebooks=sorted(declared_standalone-set(standalone_notebooks)))
        public_paths = [item[key] for item in workflows for key in ('script', 'original_notebook', 'companion')]
        public_paths += [item['notebook'] for item in json.loads(workflow_file.read_text()).get('preserved_studies', [])]
        public_paths += standalone_files
        private_references = sorted(n for n in public_paths if any(matches(n, p) for p in policy.get('private_local_globs', [])))
        check('public workflow manifest contains no private local paths', not private_references,
              private_references=private_references)
    config = (root/'tle_tailor/data/default_config.toml').read_text()
    check('distributed configuration contains no personal machine path',
          not re.search(r'/home/|/Users/|[A-Za-z]:[\\/]|Downloads/GMAT', config))
    backend = project.get('tool', {}).get('uv', {}).get('build-backend', {})
    source_excludes = backend.get('source-exclude', [])
    wheel_excludes = backend.get('wheel-exclude', [])
    private_globs = policy.get('private_local_globs', [])
    invalid_private = [p for p in private_globs if not isinstance(p, str) or not p or not safe(p)]
    # uv_build's portable glob syntax rejects literal spaces. A one-character
    # wildcard covers each space without escaping disagreements with fnmatch;
    # actual source and archive payload checks below still enforce exclusion.
    builder_private = [p.replace(' ', '?') for p in private_globs]
    uncovered_private = [p for p in builder_private if p not in source_excludes or p not in wheel_excludes]
    private_runtime = [n for n in policy['runtime_files'] if any(matches(n, p) for p in private_globs)]
    check('private local boundary is excluded by both builders and never declared runtime',
          not invalid_private and not uncovered_private and not private_runtime,
          invalid_globs=invalid_private, missing_builder_exclusions=uncovered_private,
          private_declared_runtime=private_runtime)
    denied_package_files = [n for n in source_files if excluded(n)]
    source_leaks = sorted(n for n in denied_package_files if not any(matches(n, p) for p in source_excludes))
    wheel_leaks = sorted(n for n in denied_package_files if not any(matches(n, p) for p in wheel_excludes))
    source_only_wheel_leaks = sorted(n for n in source_files if source_allowed(n) and not runtime(n)
                                   and not any(matches(n, p) for p in wheel_excludes))
    check('builder excludes every quarantined local package input from both archives',
          not source_leaks and not wheel_leaks and not source_only_wheel_leaks,
          sdist_leaks=source_leaks, wheel_leaks=wheel_leaks, source_only_wheel_leaks=source_only_wheel_leaks)
    required_runtime = [n for n in source_files if runtime(n) and not excluded(n)]
    dropped_source = sorted(n for n in required_runtime if any(matches(n, p) for p in source_excludes))
    dropped_wheel = sorted(n for n in required_runtime if any(matches(n, p) for p in wheel_excludes))
    check('builder retains every declared runtime input', not dropped_source and not dropped_wheel,
          sdist_drops=dropped_source, wheel_drops=dropped_wheel)

    def payload_check(label, payload, metadata_names):
        errors = []
        for name, data in payload.items():
            if name in metadata_names:
                continue
            allowed = runtime(name) if label == 'wheel' else source_allowed(name)
            if not safe(name) or excluded(name) or not allowed:
                errors.append({'path': name, 'reason': 'undeclared or explicitly excluded payload'})
            elif not (root/name).is_file() or data != (root/name).read_bytes():
                errors.append({'path': name, 'reason': 'artifact bytes differ from reviewed source'})
            elif name.endswith('.ipynb') and not sanitized_notebook(data):
                errors.append({'path': name, 'reason': 'notebook contains outputs/execution state'})
        required_runtime_payload = set(policy['runtime_files']) | {n for n in source_files if runtime(n) and not excluded(n)}
        absent = sorted(required_runtime_payload-set(payload))
        check(label+' payload allowlist and source byte identity', not errors and not absent,
              checked_files=len(payload), errors=errors, missing_runtime=absent)

    if wheel:
        with zipfile.ZipFile(wheel) as z:
            names = [n for n in z.namelist() if not n.endswith('/')]
            links = [item.filename for item in z.infolist()
                     if stat.S_IFMT(item.external_attr >> 16) == stat.S_IFLNK]
            check('wheel archive has safe unique regular names', len(names) == len(set(names))
                  and all(safe(n) for n in names) and not links, symlinks=links)
            infos = {n.split('/')[0] for n in names if '.dist-info/' in n}
            if len(infos) != 1:
                check('single wheel distribution metadata root', False, roots=sorted(infos))
            else:
                info = infos.pop()
                metadata = {info+'/'+n for n in ('METADATA', 'WHEEL', 'RECORD', 'entry_points.txt')}
                payload = {n: z.read(n) for n in names}
                notice_names = set()
                notice_errors = []
                for name, expected in policy['notice_sha256'].items():
                    dest = info+'/licenses/'+name
                    notice_names.add(dest)
                    if dest not in payload or sha(payload[dest]) != expected:
                        notice_errors.append(dest)
                check('wheel retains every complete reviewed notice', not notice_errors, mismatches=notice_errors)
                meta = BytesParser().parsebytes(payload[info+'/METADATA'])
                check('wheel metadata license inventory matches reviewed notices',
                      set(meta.get_all('License-File', [])) == set(policy['notice_sha256']),
                      actual=meta.get_all('License-File', []))
                check('wheel version matches declared project version', meta['Version'] == project['project']['version'],
                      actual=meta['Version'], expected=project['project']['version'])
                mandatory_gmat = [item for item in project['project'].get('dependencies', [])
                                  if re.match(r'^gmatpy(?:[<>=!~;\s]|$)', item, re.I)]
                if mandatory_gmat:
                    normalize = lambda value: re.sub(r'[\s()]', '', value).lower()
                    unconditional = [item for item in meta.get_all('Requires-Dist', []) if ';' not in item]
                    check('wheel requires pinned GMAT Py without an extra marker',
                          all(normalize(item) in {normalize(value) for value in unconditional}
                              for item in mandatory_gmat), expected=mandatory_gmat,
                          unconditional_requires_dist=unconditional)
                expected_entries = dict(project['project'].get('entry-points', {}))
                if project['project'].get('scripts'):
                    expected_entries['console_scripts'] = project['project']['scripts']
                if project['project'].get('gui-scripts'):
                    expected_entries['gui_scripts'] = project['project']['gui-scripts']
                entries = configparser.ConfigParser()
                entries.optionxform = str
                if info+'/entry_points.txt' in payload:
                    entries.read_string(payload[info+'/entry_points.txt'].decode())
                actual_entries = {section: dict(entries[section]) for section in entries.sections()}
                check('wheel entrypoints match declared project scripts exactly', actual_entries == expected_entries,
                      actual=actual_entries, expected=expected_entries)
                payload_check('wheel', payload, metadata | notice_names)
                record = list(csv.reader(io.StringIO(payload[info+'/RECORD'].decode())))
                record_errors = []
                for name, digest, size in record:
                    if not digest and name == info+'/RECORD':
                        continue
                    if name not in payload or not digest:
                        record_errors.append(name)
                        continue
                    algorithm, encoded = digest.split('=', 1)
                    actual = base64.urlsafe_b64encode(hashlib.new(algorithm, payload[name]).digest()).rstrip(b'=').decode()
                    if actual != encoded or len(payload[name]) != int(size):
                        record_errors.append(name)
                check('wheel RECORD integrity and complete inventory', not record_errors
                      and sorted(n for n, _, _ in record) == sorted(names), mismatches=record_errors)
        report['artifacts']['wheel'] = {'path': str(wheel), 'sha256': sha(wheel.read_bytes())}
    if sdist:
        with tarfile.open(sdist) as archive:
            members = archive.getmembers()
            files = [m for m in members if not m.isdir()]
            roots = {PurePosixPath(m.name).parts[0] for m in members}
            check('sdist archive has safe regular members and single root', len(roots) == 1
                  and len({m.name for m in members}) == len(members)
                  and all(safe(m.name) and m.isfile() for m in files))
            payload = {'/'.join(PurePosixPath(m.name).parts[1:]): archive.extractfile(m).read()
                       for m in files if m.isfile()}
            payload_check('sdist', payload, {'PKG-INFO'})
            required = set(policy['source_extra_files']) | set(policy['notice_sha256']) | {'distribution-policy.json', 'scripts/audit_distribution.py', 'docs/redistribution.md'}
            missing_source = sorted(required-set(payload))
            check('sdist includes notices, redistribution record and executable gate', not missing_source, missing=missing_source)
        report['artifacts']['sdist'] = {'path': str(sdist), 'sha256': sha(sdist.read_bytes())}
    report['artifact_status'] = 'never_run' if not wheel and not sdist else ('passed' if wheel and sdist else 'incomplete')
    report['status'] = 'passed' if all(c['status'] == 'passed' for c in report['checks']) else 'failed'
    if report['status'] == 'failed' and report['artifact_status'] == 'passed':
        report['artifact_status'] = 'failed'
    report['check_count'] = len(report['checks'])
    report['failure_count'] = sum(c['status'] != 'passed' for c in report['checks'])
    report['never_run'] = ['package publication', 'whole-repository legal clearance', 'provider permission verification']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--sdist', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if not (args.source_only or args.wheel or args.sdist):
        parser.error('run --source-only before building, or provide --wheel and --sdist')
    try:
        report = audit(args.root.resolve(), args.wheel, args.sdist)
    except (OSError, ValueError, KeyError, SyntaxError, tarfile.TarError, zipfile.BadZipFile) as error:
        report = {'status': 'failed', 'failure_count': 1, 'error': str(error)}
    rendered = json.dumps(report, indent=2)+'\n'
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
    print(rendered)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
