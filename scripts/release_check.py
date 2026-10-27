#!/usr/bin/env python3
"""Aggregate local candidate checks without publishing or silently waiving gates.

Default exit status describes release readiness. ``--engineering-only`` changes
only the exit policy: it never changes gate results or ``release_ready`` and
never waives an actual failed check, including a supplied artifact/receipt. Missing
release-only prerequisites remain explicit in the JSON report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {'passed', 'failed', 'blocked', 'never_run'}
REQUIRED_CASES = {'leo', 'sso', 'gps', 'geo', 'heo'}
REQUIRED_NOTEBOOK_IDS = {
    'basic_examples-tlefit_coe_fd', 'basic_examples-tlefit_coe_jax',
    'basic_examples-tlefit_eqn_fd', 'basic_examples-tlefit_eqn_jax',
    'basic_examples-tlefit_eqn_no_bstar',
    'gmat_integration-gmat_opm_prop_fd_example',
    'gmat_integration-gmat_opm_prop_jax_example',
}
TEST_FILES = (
    'test_core_functions.py', 'test_gmatpy_propagation.py', 'test_propagator_api.py',
    'test_local_gmat_propagation.py', 'test_spacetrack_auth.py',
    'test_accuracy_regression.py', 'test_accuracy_coe.py', 'test_coordinate_epochs.py',
    'test_config_portability.py', 'test_workflow_notebooks.py', 'test_no_bstar_notebook.py',
    'test_fixed_bstar_long_arc.py', 'test_mission_workflow_dispatch.py', 'test_release_check.py',
    'test_cli.py', 'test_first_use.py', 'test_examples_metadata.py', 'test_examples_layout.py',
)
ALLOWED_TEST_SKIPS = {
    'test_fixed_drag_fit_is_finite_and_improves_epoch_state':
        'Historical inline notebook is source-only; maintained fixed-BSTAR tests run separately.',
    'test_decayed_sgp4_state_is_an_explicit_fit_rejection':
        'Historical inline notebook is source-only; maintained rejection tests run separately.',
    'test_measured_gps_fixed_drag_returns_finite_improving_diagnostic_without_false_convergence':
        'Historical GPS fixture is excluded; physical fit and hold-out limits remain documented diagnostic evidence.',
    'test_native_local_gmat_point_mass_matches_independent_kepler_orbit':
        'Requires the separate, explicitly configured legacy API; not the bundled backend.',
    'test_native_local_gmat_tle_uses_independent_sgp4_initial_state':
        'Requires the separate, explicitly configured legacy API; not the bundled backend.',
    'test_source_history_inventory_and_companions_are_complete':
        'Original research notebook history is intentionally absent from distributions.',
    'test_scratch_resolver_keeps_generated_writes_out_of_preserved_inputs':
        'Preserved scratch notebook cells are source-only; public layout and synthetic path controls still run.',
    'test_scratch_file_calls_share_the_explicit_path_resolver':
        'Preserved scratch notebook cells are source-only; public layout and synthetic path controls still run.',
}


def gate(name, status, detail, *, scope='engineering', required=True, **evidence):
    if status not in STATUSES:
        raise ValueError(f'Unknown check status: {status}')
    return dict(name=name, status=status, detail=detail, scope=scope,
                required=required, **evidence)


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    value = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object receipt')
    return value


def all_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v) for v in value)
    return True


def validate_accuracy(report, expected_cases):
    """A successful partial smoke run must never qualify the release matrix."""
    if not isinstance(report, dict) or not isinstance(report.get('comparisons'), list):
        return ['accuracy report must contain a comparison list']
    expected = {(case, reference, method) for case in expected_cases
                for reference in ('sgp4', 'gmatpy') for method in ('fd', 'jax')}
    comparisons = report.get('comparisons', [])
    actual = [(item.get('case'), item.get('reference'), item.get('method'))
              for item in comparisons if isinstance(item, dict)]
    failures = []
    if set(expected_cases) != REQUIRED_CASES:
        failures.append('declared orbit cases differ from the five required release cases')
    for key in ('passed', 'complete_declared_matrix', 'release_gate_passed'):
        if report.get(key) is not True:
            failures.append(f'{key} is not true')
    if not expected or len(actual) != len(comparisons) or len(actual) != len(set(actual)) or set(actual) != expected:
        failures.append('comparison identities do not cover the complete declared matrix exactly once')
    if not comparisons or any(item.get('passed') is not True or item.get('failures') != []
                              for item in comparisons if isinstance(item, dict)):
        failures.append('missing or failed accuracy comparison')
    if not all_finite(report):
        failures.append('nonfinite values in accuracy evidence')
    return failures


def validate_coe_accuracy(report):
    """Require all classical-element cases, including independent hold-out metrics."""
    if not isinstance(report, dict) or not isinstance(report.get('comparisons'), list):
        return ['COE report must contain a comparison list']
    expected = {(case, method) for case in ('leo', 'sso', 'gps', 'geo') for method in ('fd', 'jax')}
    comparisons = report['comparisons']
    actual = [(item.get('case'), item.get('method')) for item in comparisons if isinstance(item, dict)]
    inventory = report.get('source_semantic_inventory', [])
    public_files = {'tlefit_coe_fd.py', 'tlefit_coe_jax.py', 'common_coe_fd.py', 'common_coe_jax.py'}
    problems = []
    if any(report.get(key) is not True for key in ('passed', 'complete_declared_matrix', 'release_gate_passed')):
        problems.append('COE complete release flags are not all true')
    if len(actual) != len(comparisons) or len(actual) != len(set(actual)) or set(actual) != expected:
        problems.append('COE comparison identities do not cover all eight required cases exactly once')
    if not comparisons or any(item.get('passed') is not True or item.get('failures') != []
                              for item in comparisons if isinstance(item, dict)):
        problems.append('missing or failed COE comparison')
    if (not isinstance(inventory, list) or len(inventory) != 4
        or {item.get('public') for item in inventory if isinstance(item, dict)} != public_files
        or any(not isinstance(item, dict) or item.get('same_function_asts_after_runtime_only_adapter') is not True for item in inventory)):
        problems.append('COE public/current function-equivalence evidence is incomplete or failed')
    if not all_finite(report):
        problems.append('nonfinite values in COE evidence')
    return problems


def execution_source_fingerprint(root):
    """Mirror the validator's canonical binding to actual application inputs."""
    root = Path(root).resolve()
    policy = read_json(root/'distribution-policy.json')
    names = set(policy['runtime_files']) | {'scripts/validate_notebooks.py', 'scripts/sync_workflow_notebooks.py', 'distribution-policy.json'}
    mapping = {}
    for name in sorted(names):
        path = root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
            or not path.is_file() or not path.resolve().is_relative_to(root)):
            raise ValueError(f'Execution input must be a regular project file: {name}')
        mapping[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def expected_notebooks(root):
    root = Path(root).resolve()
    manifest = read_json(root/'tle_tailor/examples/workflows.json')
    workflows = manifest['workflows']
    selected = [item for item in workflows if item['status'] == 'maintained'
                and item['input_category'].startswith('inline-')]
    if (len(selected) != 7 or {item['id'] for item in selected} != REQUIRED_NOTEBOOK_IDS
        or len({item['companion'] for item in selected}) != 7):
        raise ValueError('Notebook manifest must select all seven self-contained maintained companions exactly once')
    standalone = manifest.get('standalone_companions', [])
    primary = 'tle_tailor/examples/notebooks/deployment.ipynb'
    policy = read_json(root/'distribution-policy.json')
    if primary in policy['runtime_files'] and not any(item['notebook'] == primary for item in standalone):
        raise ValueError('Notebook manifest omits the required deployment companion')
    names = [item['companion'] for item in selected] + [item['notebook'] for item in standalone]
    if len(names) != len(set(names)):
        raise ValueError('Notebook manifest contains duplicate companions')
    result = {}
    for name in names:
        path = root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
            or not path.is_file() or not path.resolve().is_relative_to(root)):
            raise ValueError(f'Notebook must be a regular project file: {name}')
        notebook = read_json(path)
        cells = [(cell['cell_type'], ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source'])
                 for cell in notebook['cells']]
        code_indices = [index+1 for index, (kind, source) in enumerate(cells) if kind == 'code' and source.strip()]
        if not code_indices:
            raise ValueError(f'Notebook has no executable cells: {name}')
        result[name] = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'cell_source_sha256': hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest(),
                        'code_cells': code_indices}
    return result


def read_notebook_summary(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def validate_notebook_execution(records, root):
    """Require source-bound execution evidence for the entire current inventory."""
    if not isinstance(records, list) or not records or any(not isinstance(item, dict) for item in records):
        return ['Notebook execution summary must be a nonempty list of worker records']
    expected = expected_notebooks(root)
    fingerprint = execution_source_fingerprint(root)
    actual = [item.get('notebook') for item in records]
    problems = []
    if len(actual) != len(expected) or len(set(actual)) != len(expected) or set(actual) != set(expected):
        problems.append('Notebook execution does not cover the current manifest/policy inventory exactly once')
    for item in records:
        name = item.get('notebook')
        prefix = str(name)+': '
        if name not in expected:
            continue
        required = expected[name]
        if (item.get('status') != 'passed' or item.get('phase') != 'finished'
            or type(item.get('worker_returncode')) is not int or item['worker_returncode'] != 0
            or item.get('kernel_ready') is not True):
            problems.append(prefix+'fresh-kernel worker did not finish successfully')
        if (item.get('source_sha256') != required['source_sha256']
            or item.get('cell_source_sha256') != required['cell_source_sha256']
            or item.get('source_unchanged') is not True):
            problems.append(prefix+'notebook source changed or receipt is stale')
        if (item.get('execution_source_sha256') != fingerprint
            or item.get('execution_source_unchanged') is not True):
            problems.append(prefix+'application/validator/policy source changed or receipt is stale')
        cells = item.get('cells')
        if (not isinstance(cells, list) or any(not isinstance(cell, dict) for cell in cells)
            or [cell.get('cell') for cell in cells] != required['code_cells']
            or any(cell.get('status') != 'ok' for cell in cells)):
            problems.append(prefix+'one or more executable cells are missing or unsuccessful')
        opened = item.get('jupyter_load_check', {})
        if not isinstance(opened, dict) or opened.get('read') is not True or opened.get('checkpoint_created') is not True:
            problems.append(prefix+'Jupyter open/checkpoint validation is missing')
        options = item.get('validation_options', {})
        if not isinstance(options, dict) or options.get('spacetrack') is not False or options.get('catalog') is not False:
            problems.append(prefix+'offline self-contained validation options are missing or changed')
        if item.get('secret_output_detected') is not False:
            problems.append(prefix+'secret-output check did not pass')
        started, finished = item.get('started'), item.get('finished')
        if (not isinstance(started, (int, float)) or isinstance(started, bool)
            or not isinstance(finished, (int, float)) or isinstance(finished, bool)
            or started <= 0 or finished < started):
            problems.append(prefix+'execution completion timestamps are missing or invalid')
    if not all_finite(records):
        problems.append('Nonfinite notebook execution evidence')
    return problems


def validate_pytest(path):
    """Distinguish optional legacy/history skips from missing required coverage."""
    tree = ET.parse(path).getroot()
    cases = list(tree.iter('testcase'))
    failures = list(tree.iter('failure')) + list(tree.iter('error'))
    skips = [{'test': case.attrib.get('name', ''),
              'reason': case.find('skipped').attrib.get('message', '')}
             for case in cases if case.find('skipped') is not None]
    unexpected = [item for item in skips if item['test'].split('[')[0] not in ALLOWED_TEST_SKIPS]
    summary = {'tests': len(cases), 'passed': sum(case.find('skipped') is None
               and case.find('failure') is None and case.find('error') is None for case in cases),
               'failures_or_errors': len(failures), 'skipped': skips,
               'unexpected_skips': unexpected}
    problems = []
    if not cases or not summary['passed']:
        problems.append('no passing tests were collected')
    if failures:
        problems.append('pytest failures or errors')
    if unexpected:
        problems.append('required tests were skipped outside the declared optional scope')
    return problems, summary


def offline_environment(root, output):
    """Keep tests offline and away from private credentials and package inputs."""
    support = output / 'offline-support'
    support.mkdir(parents=True, exist_ok=True)
    (support / 'sitecustomize.py').write_text('''import socket
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('release checks are offline')
    return _connect(self, address)
def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('release checks are offline')
    return _connect_ex(self, address)
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
try:
    from astropy.utils import iers
except ImportError:
    pass
else:
    iers.conf.auto_download = False
''')
    environment = os.environ.copy()
    for name in ('GMAT_API_PATH', 'GMAT_PYTHON', 'SPACETRACK_USER', 'SPACETRACK_PWD', 'PYTHONHOME',
                 'TLE_TAILOR_DATA_DIR', 'TLE_TAILOR_EPHEMERIS'):
        environment.pop(name, None)
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join([str(support), str(root)]),
                       MPLBACKEND='Agg', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       TLE_TAILOR_ENV_FILE=str(output/'no-credentials.env'),
                       TLE_TAILOR_OUTPUT_DIR=str(output/'generated'),
                       MPLCONFIGDIR=str(output/'cache/matplotlib'), XDG_CACHE_HOME=str(output/'cache'))
    return environment


def run_command(name, command, root, output, environment, timeout):
    started = time.monotonic()
    log = output / f'{name}.log'
    with log.open('w') as stream:
        try:
            child = subprocess.run(command, cwd=root, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {'command': command, 'log': str(log), 'returncode': None,
                    'duration_s': time.monotonic()-started, 'error': f'{type(error).__name__}: {error}'}
    return {'command': command, 'log': str(log), 'returncode': child.returncode,
            'duration_s': time.monotonic()-started}


def conclusions(gates, engineering_only=False):
    required = [item for item in gates if item['required']]
    engineering = [item for item in required if item['scope'] == 'engineering']
    engineering_passed = (bool(engineering) and all(item['status'] == 'passed' for item in engineering)
                          and not any(item['status'] == 'failed' for item in required))
    release_ready = bool(required) and all(item['status'] == 'passed' for item in required)
    return {'engineering_passed': engineering_passed, 'release_ready': release_ready,
            'exit_status': 0 if (engineering_passed if engineering_only else release_ready) else 1,
            'exit_policy': 'engineering-only; release readiness is still reported' if engineering_only else 'all required release gates'}


def declared_gmatpy_version(project):
    requirements = project.get('dependencies', [])
    if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
        raise ValueError('project.dependencies must be a list of requirement strings')
    try:
        matches = [parsed for text in requirements if canonicalize_name((parsed := Requirement(text)).name) == 'gmatpy']
    except ValueError as error:
        raise ValueError('Invalid project dependency requirement: ' + str(error)) from error
    if len(matches) != 1:
        raise ValueError('project.dependencies must declare exactly one gmatpy requirement')
    requirement = matches[0]
    pins = list(requirement.specifier)
    if (requirement.marker is not None or requirement.url or requirement.extras or len(pins) != 1
            or pins[0].operator != '==' or '*' in pins[0].version):
        raise ValueError('gmatpy dependency must be an unconditional exact == version')
    return pins[0].version


def validate_install_receipt(receipt, wheel_sha256, version, dependency_version):
    expected = {'schema_version': 1, 'status': 'passed', 'outside_checkout': True,
                'project_version': version, 'gmatpy_version': dependency_version, 'wheel_sha256': wheel_sha256}
    return [f'{key} does not match the qualified artifact/environment' for key, value in expected.items()
            if receipt.get(key) != value]


def run_checks(args):
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = offline_environment(root, output)
    results = []
    source_report = output/'distribution-source.json'
    source_report.unlink(missing_ok=True)
    source_run = run_command('distribution-source', [sys.executable, 'scripts/audit_distribution.py',
        '--source-only', '--json', str(source_report)], root, output, environment, args.timeout)
    source = {}
    try:
        source = read_json(source_report)
        source_passed = source_run['returncode'] == 0 and source.get('status') == 'passed' and source.get('failure_count') == 0
        results.append(gate('distribution-source', 'passed' if source_passed else 'failed',
                            'Source payload boundary and notices checked; this is not an artifact or whole-repository clearance.', **source_run))
    except (OSError, ValueError) as error:
        results.append(gate('distribution-source', 'failed', f'Missing/invalid source audit: {error}', **source_run))

    sync_run = run_command('notebook-parity', [sys.executable, 'scripts/sync_workflow_notebooks.py', '--check', '--distributed'],
                           root, output, environment, args.timeout)
    results.append(gate('notebook-parity', 'passed' if sync_run['returncode'] == 0 else 'failed',
                        'Checks maintained scripts/companions; original source-history notebooks are not executed by this gate.', **sync_run))

    notebook_report = getattr(args, 'notebook_execution_report', None)
    if notebook_report is None:
        results.append(gate('fresh-kernel-notebook-execution', 'never_run',
                            'No fresh-kernel receipt supplied; parity does not establish working notebooks.', scope='release'))
    else:
        try:
            notebook_records = read_notebook_summary(notebook_report)
            problems = validate_notebook_execution(notebook_records, root)
            command_status = getattr(args, 'notebook_command_status', None)
            if command_status is not None and command_status != 'success':
                problems.append('The notebook validator command did not complete successfully')
            results.append(gate('fresh-kernel-notebook-execution', 'passed' if not problems else 'failed',
                                '; '.join(problems) if problems else 'All inventory-selected self-contained companions executed every code cell in fresh kernels; source preserved.',
                                scope='release', report=str(notebook_report.resolve()), notebooks=len(notebook_records) if isinstance(notebook_records, list) else 0))
        except (OSError, ValueError, TypeError, KeyError) as error:
            results.append(gate('fresh-kernel-notebook-execution', 'failed', f'Missing/invalid notebook execution evidence: {error}', scope='release'))

    test_paths = [f'tle_tailor/tests/{name}' for name in TEST_FILES]
    missing = [name for name in test_paths if not (root/name).is_file()]
    junit = output/'tests.xml'
    junit.unlink(missing_ok=True)
    if missing:
        results.append(gate('self-contained-tests', 'blocked', 'Required focused test files are missing.', missing=missing, execution_status='never_run'))
    else:
        test_run = run_command('tests', [sys.executable, '-m', 'pytest', *test_paths, '-q', '--junitxml='+str(junit),
                               '-o', 'cache_dir='+str(output/'pytest-cache')], root, output, environment, args.timeout)
        try:
            problems, test_summary = validate_pytest(junit)
            if test_run['returncode'] != 0:
                problems.append('pytest command did not exit successfully')
            results.append(gate('self-contained-tests', 'passed' if not problems else 'failed',
                                '; '.join(problems) if problems else 'All focused tests passed; optional skips listed individually.',
                                summary=test_summary, test_files=test_paths, **test_run))
        except (OSError, ValueError, ET.ParseError) as error:
            results.append(gate('self-contained-tests', 'failed', f'Missing/invalid pytest evidence: {error}', **test_run))

    accuracy_dir = output/'accuracy'
    accuracy_dir.mkdir(exist_ok=True)
    accuracy_path = accuracy_dir/'accuracy-report.json'
    accuracy_path.unlink(missing_ok=True)
    accuracy_run = run_command('accuracy', [sys.executable, 'scripts/accuracy_regression.py', '--output', str(accuracy_dir)],
                               root, output, environment, args.timeout)
    try:
        cases = {item['id'] for item in read_json(root/'tle_tailor/validation/accuracy_cases.json')['cases']}
        accuracy = read_json(accuracy_path)
        problems = validate_accuracy(accuracy, cases)
        if accuracy_run['returncode'] != 0:
            problems.append('accuracy command did not exit successfully')
        results.append(gate('public-fit-holdout-accuracy', 'passed' if not problems else 'failed',
                            '; '.join(problems) if problems else 'Complete public/candidate matrix passed with unchanged no-regression criteria.',
                            report=str(accuracy_path), comparisons=len(accuracy.get('comparisons', [])), **accuracy_run))
    except (OSError, ValueError, KeyError, TypeError) as error:
        results.append(gate('public-fit-holdout-accuracy', 'failed', f'Missing/invalid complete accuracy evidence: {error}', **accuracy_run))

    coe_dir = output/'coe-accuracy'
    coe_dir.mkdir(exist_ok=True)
    coe_path = coe_dir/'coe-accuracy-report.json'
    coe_path.unlink(missing_ok=True)
    coe_run = run_command('coe-accuracy', [sys.executable, 'scripts/coe_accuracy_comparison.py', '--output', str(coe_dir)],
                          root, output, environment, args.timeout)
    try:
        coe_report = read_json(coe_path)
        problems = validate_coe_accuracy(coe_report)
        if coe_run['returncode'] != 0:
            problems.append('COE comparison command did not exit successfully')
        results.append(gate('public-coe-fit-holdout-accuracy', 'passed' if not problems else 'failed',
                            '; '.join(problems) if problems else 'All eight classical-element public/candidate comparisons passed.',
                            report=str(coe_path), comparisons=len(coe_report.get('comparisons', [])), **coe_run))
    except (OSError, ValueError, KeyError, TypeError) as error:
        results.append(gate('public-coe-fit-holdout-accuracy', 'failed', f'Missing/invalid complete COE evidence: {error}', **coe_run))

    if args.wheel is None or args.sdist is None:
        results.append(gate('distribution-artifacts', 'blocked', 'Provide both --wheel and --sdist; source preflight is insufficient.',
                            scope='release', execution_status='never_run'))
    else:
        artifact_report = output/'distribution-artifacts.json'
        artifact_report.unlink(missing_ok=True)
        artifact_run = run_command('distribution-artifacts', [sys.executable, 'scripts/audit_distribution.py',
            '--wheel', str(args.wheel.resolve()), '--sdist', str(args.sdist.resolve()), '--json', str(artifact_report)],
            root, output, environment, args.timeout)
        try:
            artifact = read_json(artifact_report)
            passed = artifact_run['returncode'] == 0 and artifact.get('status') == 'passed' and artifact.get('artifact_status') == 'passed'
            results.append(gate('distribution-artifacts', 'passed' if passed else 'failed',
                                'Both built archives checked against the explicit payload policy.', scope='release', **artifact_run))
        except (OSError, ValueError) as error:
            results.append(gate('distribution-artifacts', 'failed', f'Missing/invalid built artifact audit: {error}', scope='release', **artifact_run))

    project = tomllib.loads((root/'pyproject.toml').read_text())['project']
    version = project['version']
    if args.external_install_report is None:
        results.append(gate('clean-external-install', 'never_run', 'No clean installation receipt supplied.', scope='release'))
    else:
        try:
            if args.wheel is None:
                raise ValueError('The tested wheel is required to bind installation evidence to artifact bytes.')
            digest = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
            receipt = read_json(args.external_install_report)
            problems = validate_install_receipt(receipt, digest, version, declared_gmatpy_version(project))
            results.append(gate('clean-external-install', 'passed' if not problems else 'failed',
                                '; '.join(problems) if problems else 'External installation receipt matches this wheel and published GMAT dependency.',
                                scope='release', report=str(args.external_install_report.resolve())))
        except (OSError, ValueError) as error:
            results.append(gate('clean-external-install', 'failed', f'Invalid installation evidence: {error}', scope='release'))

    if args.legacy_engine_report is None:
        results.append(gate('legacy-native-engine-accuracy', 'blocked',
                            'No independent matched-public-engine accuracy receipt supplied; common-truth fitting does not establish native-engine parity.',
                            scope='release', execution_status='never_run'))
    else:
        try:
            receipt = read_json(args.legacy_engine_report)
            passed = receipt.get('status') == 'passed' and receipt.get('release_gate_passed') is True
            status = 'passed' if passed else ('failed' if receipt.get('status') == 'failed' else 'blocked')
            results.append(gate('legacy-native-engine-accuracy', status,
                                'Matched public-engine/workflow no-regression comparison passed.' if passed else
                                'Matched public-engine/workflow comparison has not passed; inspect the supplied receipt for measured differences and remaining gaps.',
                                scope='release', report=str(args.legacy_engine_report.resolve()),
                                blockers=receipt.get('blockers', []), execution_status=receipt.get('execution_status', 'unknown')))
        except (OSError, ValueError) as error:
            results.append(gate('legacy-native-engine-accuracy', 'blocked', f'Missing/invalid independent engine receipt: {error}', scope='release'))

    repo_passed = source.get('repository_publication_status') == 'passed'
    results.append(gate('repository-redistribution', 'passed' if repo_passed else 'blocked',
                        'Whole-repository publication provenance is separately checked.' if repo_passed else
                        'The complete development checkout/history is outside the curated source-release boundary; artifact and curated-tree audits remain required.', scope='context', required=False,
                        blockers=source.get('repository_publication_blockers', ['Source audit does not establish repository clearance.'])))
    try:
        workflow = read_json(root/'tle_tailor/examples/workflows.json')
        diagnostics = [item for item in workflow['workflows'] if item['status'] == 'diagnostic']
        results.append(gate('diagnostic-fit-qualification', 'blocked' if diagnostics else 'passed',
                            'Diagnostic examples retain their stopping flags and reported fit/hold-out limits; they are not claims of general convergence or forecast accuracy.' if diagnostics else 'No diagnostic fit is advertised.',
                            scope='context', required=False, diagnostics=[dict(id=item['id'], detail=item.get('original_differences', [])) for item in diagnostics]))
    except (OSError, ValueError, KeyError) as error:
        results.append(gate('diagnostic-fit-qualification', 'blocked', f'Cannot inspect diagnostic qualification: {error}', scope='release'))
    results.extend([
        gate('hosted-ci', 'never_run', 'This local command does not dispatch or establish a hosted CI run.', scope='context', required=False),
        gate('publication', 'never_run', 'Publication requires separate approval and is not performed by this command.', scope='context', required=False),
        gate('full-source-history-suite', 'never_run', 'Focused self-contained tests do not claim execution of every private-data/research notebook or script.', scope='context', required=False),
    ])
    return dict(schema_version=1, started_from=str(root), qualified_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, mode='engineering-only' if args.engineering_only else 'release',
                checks=results, **conclusions(results, args.engineering_only))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('output/release-check'))
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--sdist', type=Path)
    parser.add_argument('--external-install-report', type=Path)
    parser.add_argument('--legacy-engine-report', type=Path)
    parser.add_argument('--notebook-execution-report', type=Path, help='fresh-kernel validator summary.json for all inventory-selected self-contained companions')
    parser.add_argument('--notebook-command-status', choices=['success', 'failure', 'cancelled', 'skipped'], help='optional CI command outcome; a failed command cannot be waived by an old receipt')
    parser.add_argument('--engineering-only', action='store_true', help='change exit policy only; never waive accuracy or claim release readiness')
    parser.add_argument('--timeout', type=float, default=1800, help='maximum seconds for each child check')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'release-check.json').unlink(missing_ok=True)
    report = run_checks(args)
    path = args.output/'release-check.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for item in report['checks']:
        print(f"{item['name']}: {item['status'].upper()} — {item['detail']}")
    print(f"Engineering passed: {report['engineering_passed']}; release ready: {report['release_ready']}; {path}")
    return report['exit_status']


if __name__ == '__main__':
    raise SystemExit(main())
