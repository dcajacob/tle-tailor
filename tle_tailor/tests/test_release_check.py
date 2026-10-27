"""Adversarial release decisions: successful commands must not hide missing gates."""
from copy import deepcopy
import hashlib
import time
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('release_check_under_test', ROOT/'scripts/release_check.py')
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)
CLEAN_SPEC = importlib.util.spec_from_file_location('verify_clean_install_under_test', ROOT/'scripts/verify_clean_install.py')
CLEAN = importlib.util.module_from_spec(CLEAN_SPEC)
CLEAN_SPEC.loader.exec_module(CLEAN)


def complete_accuracy():
    return {'passed': True, 'complete_declared_matrix': True, 'release_gate_passed': True,
            'comparisons': [dict(case=case, reference=reference, method=method, passed=True, failures=[],
                                 independently_measured_error=0.001)
                            for case in sorted(RELEASE.REQUIRED_CASES)
                            for reference in ('sgp4', 'gmatpy') for method in ('fd', 'jax')]}


def complete_coe_accuracy():
    return {'passed': True, 'complete_declared_matrix': True, 'release_gate_passed': True,
            'comparisons': [dict(case=case, method=method, passed=True, failures=[])
                            for case in ('leo', 'sso', 'gps', 'geo') for method in ('fd', 'jax')],
            'source_semantic_inventory': [dict(public=name, same_function_asts_after_runtime_only_adapter=True)
                for name in ('tlefit_coe_fd.py', 'tlefit_coe_jax.py', 'common_coe_fd.py', 'common_coe_jax.py')]}


@pytest.mark.parametrize('mutation', ['partial', 'missing-flag', 'failed-holdout', 'source-mismatch', 'missing-source'])
def test_coe_partial_or_failed_evidence_is_mandatory(mutation):
    report = complete_coe_accuracy()
    assert RELEASE.validate_coe_accuracy(report) == []
    if mutation == 'partial':
        report['comparisons'] = report['comparisons'][:2]
    elif mutation == 'missing-flag':
        del report['release_gate_passed']
    elif mutation == 'failed-holdout':
        report['comparisons'][0].update(passed=False, failures=['hold-out regressed'])
    elif mutation == 'source-mismatch':
        report['source_semantic_inventory'][0]['same_function_asts_after_runtime_only_adapter'] = False
    else:
        report['source_semantic_inventory'] = []
    assert RELEASE.validate_coe_accuracy(report)


def test_complete_declared_accuracy_matrix_is_required():
    report = complete_accuracy()
    assert RELEASE.validate_accuracy(report, RELEASE.REQUIRED_CASES) == []
    report['comparisons'] = report['comparisons'][:4]
    assert RELEASE.validate_accuracy(report, RELEASE.REQUIRED_CASES)
    # Even deliberately reducing the input manifest cannot qualify a smoke run.
    assert RELEASE.validate_accuracy(report, {report['comparisons'][0]['case']})


@pytest.mark.parametrize('flag', ['passed', 'complete_declared_matrix', 'release_gate_passed'])
def test_report_flags_cannot_be_missing_or_false(flag):
    report = complete_accuracy()
    del report[flag]
    assert RELEASE.validate_accuracy(report, RELEASE.REQUIRED_CASES)
    report[flag] = False
    assert RELEASE.validate_accuracy(report, RELEASE.REQUIRED_CASES)


@pytest.mark.parametrize('mutation', ['duplicate', 'failed', 'nonfinite', 'wrong-reference'])
def test_individual_matrix_failures_cannot_be_hidden_by_top_level_success(mutation):
    report = complete_accuracy()
    if mutation == 'duplicate':
        report['comparisons'][1] = deepcopy(report['comparisons'][0])
    elif mutation == 'failed':
        report['comparisons'][0].update(passed=False, failures=['hold-out position regressed'])
    elif mutation == 'nonfinite':
        report['comparisons'][0]['independently_measured_error'] = float('nan')
    else:
        report['comparisons'][0]['reference'] = 'unverified-new-truth'
    assert RELEASE.validate_accuracy(report, RELEASE.REQUIRED_CASES)


def test_empty_or_wrong_type_report_does_not_pass():
    for report in ({}, [], {'comparisons': []}, {'comparisons': 'success'}):
        assert RELEASE.validate_accuracy(report, RELEASE.REQUIRED_CASES)


def test_nonfinite_json_is_rejected_before_decision(tmp_path):
    receipt = tmp_path/'accuracy.json'
    receipt.write_text('{"passed": true, "metric": NaN}')
    with pytest.raises(ValueError, match='Nonfinite'):
        RELEASE.read_json(receipt)


@pytest.mark.parametrize('engineering_only', [False, True])
def test_accuracy_failure_is_never_waived(engineering_only):
    checks = [RELEASE.gate('unit-tests', 'passed', 'test fixture'),
              RELEASE.gate('accuracy', 'failed', 'hold-out regression'),
              RELEASE.gate('publication-policy', 'blocked', 'unreviewed inputs', scope='release')]
    result = RELEASE.conclusions(checks, engineering_only)
    assert result['exit_status'] == 1
    assert result['engineering_passed'] is False and result['release_ready'] is False


def test_engineering_exit_policy_does_not_relabel_release_blocks():
    checks = [RELEASE.gate('accuracy', 'passed', 'complete strict matrix'),
              RELEASE.gate('artifacts', 'blocked', 'not built', scope='release'),
              RELEASE.gate('external-install', 'never_run', 'no receipt', scope='release')]
    assert RELEASE.conclusions(checks)['exit_status'] == 1
    result = RELEASE.conclusions(checks, engineering_only=True)
    assert result['exit_status'] == 0 and result['release_ready'] is False
    assert [item['status'] for item in checks] == ['passed', 'blocked', 'never_run']


@pytest.mark.parametrize('contents', [
    '<testsuite tests="0"/>',
    '<testsuite><testcase name="test_native_frame"><skipped message="native wheel absent"/></testcase></testsuite>',
    '<testsuite><testcase name="test_fit"><failure>bad holdout</failure></testcase></testsuite>',
])
def test_missing_or_skipped_required_test_coverage_blocks(tmp_path, contents):
    receipt = tmp_path/'tests.xml'; receipt.write_text(contents)
    problems, _ = RELEASE.validate_pytest(receipt)
    assert problems


def test_optional_skips_remain_visible_without_waiving_required_tests(tmp_path):
    receipt = tmp_path/'tests.xml'
    receipt.write_text('<testsuite><testcase name="test_required_fit"/>'
                      '<testcase name="test_native_local_gmat_point_mass_matches_independent_kepler_orbit">'
                      '<skipped message="separate API unavailable"/></testcase></testsuite>')
    problems, summary = RELEASE.validate_pytest(receipt)
    assert not problems
    assert summary['passed'] == 1 and len(summary['skipped']) == 1


def test_clean_install_is_bound_to_exact_wheel_not_just_version():
    receipt = dict(schema_version=1, status='passed', outside_checkout=True,
                   project_version='0.1.0rc1', gmatpy_version='0.2.1', wheel_sha256='old-artifact')
    assert RELEASE.validate_install_receipt(receipt, 'new-artifact', '0.1.0rc1', '0.2.1')
    receipt['wheel_sha256'] = 'new-artifact'
    assert not RELEASE.validate_install_receipt(receipt, 'new-artifact', '0.1.0rc1', '0.2.1')
    receipt['outside_checkout'] = False
    assert RELEASE.validate_install_receipt(receipt, 'new-artifact', '0.1.0rc1', '0.2.1')


def test_clean_install_receipt_requires_explicit_current_dependency_version():
    receipt = dict(schema_version=1, status='passed', outside_checkout=True,
                   project_version='0.1.0rc2', gmatpy_version='0.2.2', wheel_sha256='artifact')
    assert not RELEASE.validate_install_receipt(receipt, 'artifact', '0.1.0rc2', '0.2.2')
    assert RELEASE.validate_install_receipt(receipt, 'artifact', '0.1.0rc2', '0.2.1')
    assert RELEASE.validate_install_receipt(receipt, 'artifact', '0.1.0rc1', '0.2.2')


def test_declared_gmatpy_version_comes_from_exact_project_pin():
    assert RELEASE.declared_gmatpy_version({'dependencies': ['numpy>=1', 'GMATPy == 0.2.2']}) == '0.2.2'


@pytest.mark.parametrize('requirements', [None, [], ['gmatpy>=0.2.1'], ['gmatpy==0.2.*'],
    ['gmatpy==0.2.1', 'gmatpy==0.2.2'], ['gmatpy==0.2.2; sys_platform == "win32"'],
    ['gmatpy[extra]==0.2.2'], ['gmatpy @ https://example.invalid/gmatpy.whl'], ['not a requirement']])
def test_declared_gmatpy_version_rejects_ambiguous_or_invalid_requirements(requirements):
    with pytest.raises(ValueError):
        RELEASE.declared_gmatpy_version({'dependencies': requirements})


def write_metadata_wheel(path, name, version, requirements=()):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('test.dist-info/METADATA',
            f'Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n' +
            ''.join('Requires-Dist: ' + item + '\n' for item in requirements))
    return path


def test_clean_install_versions_are_read_from_both_supplied_wheels(tmp_path):
    app = write_metadata_wheel(tmp_path/'app.whl', 'tle_tailor', '0.1.0rc2',
                               ['gmatpy==0.2.2', 'gmatpy==0.2.2; extra == "gmat"'])
    dependency = write_metadata_wheel(tmp_path/'dep.whl', 'gmatpy', '0.2.2')
    assert CLEAN.expected_wheel_versions(app, dependency) == {'tle-tailor': '0.1.0rc2', 'gmatpy': '0.2.2'}
    write_metadata_wheel(dependency, 'gmatpy', '0.2.1')
    with pytest.raises(ValueError, match='differs from the application dependency pin'):
        CLEAN.expected_wheel_versions(app, dependency)


@pytest.mark.parametrize('requirements', [[], ['gmatpy>=0.2.1'], ['gmatpy==0.2.*'],
    ['gmatpy==0.2.2', 'gmatpy==0.2.2'], ['invalid requirement']])
def test_clean_install_rejects_missing_unpinned_or_invalid_wheel_dependency(tmp_path, requirements):
    app = write_metadata_wheel(tmp_path/'app.whl', 'tle-tailor', '0.1.0rc2', requirements)
    dependency = write_metadata_wheel(tmp_path/'dep.whl', 'gmatpy', '0.2.2')
    with pytest.raises(ValueError):
        CLEAN.expected_wheel_versions(app, dependency)


def test_clean_install_rejects_mislabeled_or_incomplete_wheel_metadata(tmp_path):
    wheel = write_metadata_wheel(tmp_path/'app.whl', 'different-project', '0.1.0rc2')
    with pytest.raises(ValueError, match='project name'):
        CLEAN.wheel_metadata(wheel, 'tle-tailor')
    write_metadata_wheel(wheel, 'tle-tailor', '')
    with pytest.raises(ValueError, match='Missing wheel Version'):
        CLEAN.wheel_metadata(wheel, 'tle-tailor')
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr('second.dist-info/METADATA', 'Name: tle-tailor\nVersion: 0.1.0rc2\n')
    with pytest.raises(ValueError, match='exactly one'):
        CLEAN.wheel_metadata(wheel, 'tle-tailor')


@pytest.fixture
def simulated_gate_commands(tmp_path, monkeypatch):
    root = tmp_path/'source'; root.mkdir()
    output = tmp_path/'evidence'
    (root/'pyproject.toml').write_text('[project]\nversion="0.1.0rc1"\n')
    for name in RELEASE.TEST_FILES:
        path = root/'tle_tailor/tests'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('')
    manifest = root/'tle_tailor/validation/accuracy_cases.json'; manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'cases': [{'id': case} for case in RELEASE.REQUIRED_CASES]}))
    workflow = root/'tle_tailor/examples/workflows.json'; workflow.parent.mkdir(parents=True)
    workflow.write_text('{"workflows": []}')
    controls = {'accuracy': complete_accuracy(), 'coe_accuracy': complete_coe_accuracy(), 'test_exit': 0, 'write_accuracy': True}

    def fake_run(name, command, source, evidence, environment, timeout):
        if name == 'distribution-source':
            Path(command[-1]).write_text(json.dumps({'status': 'passed', 'failure_count': 0,
                 'repository_publication_status': 'blocked', 'repository_publication_blockers': ['review needed']}))
        if name == 'tests':
            path = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
            path.write_text('<testsuite><testcase name="test_required_accuracy"/></testsuite>')
        if name == 'accuracy' and controls['write_accuracy']:
            path = Path(command[-1])/'accuracy-report.json'
            path.write_text(json.dumps(controls['accuracy']))
        if name == 'coe-accuracy':
            (Path(command[-1])/'coe-accuracy-report.json').write_text(json.dumps(controls['coe_accuracy']))
        return {'returncode': controls['test_exit'] if name == 'tests' else 0,
                'command': command, 'log': str(evidence/f'{name}.log'), 'duration_s': 0.0}

    monkeypatch.setattr(RELEASE, 'run_command', fake_run)
    args = SimpleNamespace(root=root, output=output, timeout=1, engineering_only=False,
                           wheel=None, sdist=None, external_install_report=None, legacy_engine_report=None)
    return args, controls


def test_successful_children_do_not_imply_artifacts_or_release_ready(simulated_gate_commands):
    args, _ = simulated_gate_commands
    report = RELEASE.run_checks(args)
    statuses = {item['name']: item['status'] for item in report['checks']}
    assert report['engineering_passed'] is True and report['release_ready'] is False
    assert report['exit_status'] == 1
    assert statuses['distribution-artifacts'] == 'blocked'
    assert statuses['clean-external-install'] == 'never_run'
    assert statuses['legacy-native-engine-accuracy'] == 'blocked'
    assert statuses['repository-redistribution'] == 'blocked'


def test_partial_successful_accuracy_child_still_fails_aggregate(simulated_gate_commands):
    args, controls = simulated_gate_commands
    controls['accuracy']['comparisons'] = controls['accuracy']['comparisons'][:4]
    args.engineering_only = True
    report = RELEASE.run_checks(args)
    assert report['engineering_passed'] is False
    assert report['exit_status'] == 1 and report['release_ready'] is False


def test_nonzero_tests_cannot_be_overridden_by_successful_junit(simulated_gate_commands):
    args, controls = simulated_gate_commands
    controls['test_exit'] = 1
    args.engineering_only = True
    report = RELEASE.run_checks(args)
    assert report['exit_status'] == 1
    assert next(item for item in report['checks'] if item['name'] == 'self-contained-tests')['status'] == 'failed'


def test_stale_accuracy_receipt_does_not_survive_missing_new_output(simulated_gate_commands):
    args, controls = simulated_gate_commands
    path = args.output/'accuracy/accuracy-report.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(complete_accuracy()))
    controls['write_accuracy'] = False
    report = RELEASE.run_checks(args)
    assert report['exit_status'] == 1
    assert next(item for item in report['checks'] if item['name'] == 'public-fit-holdout-accuracy')['status'] == 'failed'


def test_missing_full_coe_matrix_cannot_be_waived_by_engineering_mode(simulated_gate_commands):
    args, controls = simulated_gate_commands
    controls['coe_accuracy']['comparisons'] = controls['coe_accuracy']['comparisons'][:2]
    args.engineering_only = True
    report = RELEASE.run_checks(args)
    assert report['exit_status'] == 1 and report['engineering_passed'] is False
    assert next(item for item in report['checks'] if item['name'] == 'public-coe-fit-holdout-accuracy')['status'] == 'failed'


@pytest.mark.parametrize('contents', ['null', '[]', 'true', '"success"'])
def test_non_object_receipts_are_rejected_instead_of_crashing_mid_gate(tmp_path, contents):
    receipt = tmp_path/'receipt.json'; receipt.write_text(contents)
    with pytest.raises(ValueError, match='JSON object'):
        RELEASE.read_json(receipt)


def test_engineering_mode_does_not_ignore_a_built_artifact_leak():
    checks = [RELEASE.gate('accuracy', 'passed', 'complete strict matrix'),
              RELEASE.gate('distribution-artifacts', 'failed', 'private BSP leaked', scope='release')]
    report = RELEASE.conclusions(checks, engineering_only=True)
    assert report['exit_status'] == 1
    assert report['engineering_passed'] is False and report['release_ready'] is False


def notebook_receipts_for_source(root):
    """Build unit-test metadata from source; this never executes a notebook."""
    expected = RELEASE.expected_notebooks(root)
    fingerprint = RELEASE.execution_source_fingerprint(root)
    started = time.time()
    return [dict(notebook=name, status='passed', phase='finished', source_sha256=item['source_sha256'],
        cell_source_sha256=item['cell_source_sha256'], source_unchanged=True,
        execution_source_sha256=fingerprint, execution_source_unchanged=True, kernel_ready=True,
        worker_returncode=0, cells=[dict(cell=index, started=started, elapsed_s=0.01, status='ok') for index in item['code_cells']],
        started=started, finished=started+1, jupyter_load_check=dict(read=True, checkpoint_created=True),
        validation_options=dict(spacetrack=False, catalog=False), secret_output_detected=False)
        for name, item in expected.items()]


@pytest.fixture
def executed_notebook_receipts(tmp_path):
    """The current seven workflows plus deployment, with synthetic worker metadata."""
    root = tmp_path/'notebook-source'; root.mkdir()
    script = root/'scripts/validate_notebooks.py'; script.parent.mkdir(); script.write_text('# validator fixture\n')
    (root/'scripts/sync_workflow_notebooks.py').write_text('# inventory fixture\n')
    module = root/'tle_tailor/helper.py'; module.parent.mkdir(); module.write_text('VALUE = 1\n')
    entries = []
    for identifier in sorted(RELEASE.REQUIRED_NOTEBOOK_IDS):
        folder, name = identifier.split('-', 1)
        companion = f'tle_tailor/examples/notebooks/maintained/{folder}/{name}.ipynb'
        path = root/companion; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'cells': [
            {'cell_type': 'markdown', 'source': ['Offline companion']},
            {'cell_type': 'code', 'source': ['print(1)'], 'outputs': [], 'execution_count': None},
        ], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}))
        entries.append(dict(id=identifier, companion=companion, status='maintained', input_category='inline-tle'))
    manifest = root/'tle_tailor/examples/workflows.json'
    primary = 'tle_tailor/examples/notebooks/deployment.ipynb'
    (root/primary).write_text(json.dumps({'cells': [
        {'cell_type': 'markdown', 'source': ['Deployment companion']},
        {'cell_type': 'code', 'source': ['print("deployment")'], 'outputs': [], 'execution_count': None},
    ], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}))
    manifest.write_text(json.dumps({'workflows': entries, 'standalone_companions': [{'notebook': primary}]}))
    (root/'distribution-policy.json').write_text(json.dumps({'runtime_files': [
        'tle_tailor/helper.py', 'tle_tailor/examples/workflows.json', primary]}))
    return root, notebook_receipts_for_source(root)


def test_entire_manifest_inventory_requires_source_bound_kernel_receipts(executed_notebook_receipts):
    root, records = executed_notebook_receipts
    assert RELEASE.validate_notebook_execution(records, root) == []
    assert RELEASE.validate_notebook_execution(records[:1], root)
    assert RELEASE.validate_notebook_execution([], root)


@pytest.fixture
def current_public_notebook_receipts():
    """Actual packaged manifest/policy/notebook bytes; synthetic execution metadata."""
    manifest = json.loads((ROOT/'tle_tailor/examples/workflows.json').read_text())
    primary = 'tle_tailor/examples/notebooks/deployment.ipynb'
    declared = {item['companion'] for item in manifest['workflows']
                if item['status'] == 'maintained' and item['input_category'].startswith('inline-')}
    declared.update(item['notebook'] for item in manifest['standalone_companions'])
    assert len(declared) == 8 and primary in declared
    records = notebook_receipts_for_source(ROOT)
    assert {record['notebook'] for record in records} == declared
    return ROOT, records


def test_actual_current_eight_notebook_inventory_accepts_complete_receipts(current_public_notebook_receipts):
    root, records = current_public_notebook_receipts
    assert RELEASE.validate_notebook_execution(records, root) == []


@pytest.mark.parametrize('mutation', ['missing-primary', 'missing-workflow', 'duplicate-replacement',
    'duplicate-extra', 'unexpected-replacement', 'unexpected-extra'])
def test_current_inventory_rejects_missing_duplicate_or_unexpected_receipts(current_public_notebook_receipts, mutation):
    root, records = current_public_notebook_receipts
    primary = 'tle_tailor/examples/notebooks/deployment.ipynb'
    if mutation == 'missing-primary':
        records = [record for record in records if record['notebook'] != primary]
    elif mutation == 'missing-workflow':
        records = [record for record in records if record['notebook'] != records[0]['notebook']]
    elif mutation == 'duplicate-replacement':
        records[1] = deepcopy(records[0])
    elif mutation == 'duplicate-extra':
        records.append(deepcopy(records[0]))
    else:
        unexpected = dict(deepcopy(records[0]), notebook='tle_tailor/examples/notebooks/unexpected.ipynb')
        if mutation == 'unexpected-replacement':
            records[0] = unexpected
        else:
            records.append(unexpected)
    assert RELEASE.validate_notebook_execution(records, root) == [
        'Notebook execution does not cover the current manifest/policy inventory exactly once']


def test_receipt_count_tracks_an_additional_manifest_and_policy_standalone(executed_notebook_receipts):
    root, _ = executed_notebook_receipts
    extra = 'tle_tailor/examples/notebooks/additional.ipynb'
    (root/extra).write_bytes((root/'tle_tailor/examples/notebooks/deployment.ipynb').read_bytes())
    path = root/'tle_tailor/examples/workflows.json'; manifest = json.loads(path.read_text())
    manifest['standalone_companions'].append({'notebook': extra}); path.write_text(json.dumps(manifest))
    path = root/'distribution-policy.json'; policy = json.loads(path.read_text())
    policy['runtime_files'].append(extra); path.write_text(json.dumps(policy))
    records = notebook_receipts_for_source(root)
    assert len(records) == 9
    assert RELEASE.validate_notebook_execution(records, root) == []


@pytest.mark.parametrize('mutation', ['duplicate', 'running', 'no-cells', 'cell-error', 'changed-source',
    'changed-application', 'source-flag-false', 'runtime-flag-false', 'no-kernel', 'failed-worker',
    'missing-finish', 'live-options', 'missing-open-check'])
def test_notebook_receipt_failures_cannot_be_hidden_by_passed_status(executed_notebook_receipts, mutation):
    root, records = executed_notebook_receipts
    if mutation == 'duplicate':
        records[1] = deepcopy(records[0])
    elif mutation == 'running':
        records[0]['status'] = 'running'
    elif mutation == 'no-cells':
        records[0]['cells'] = []
    elif mutation == 'cell-error':
        records[0]['cells'][0]['status'] = 'error'
    elif mutation == 'changed-source':
        path = root/records[0]['notebook']; path.write_text(path.read_text()+'\n')
    elif mutation == 'changed-application':
        (root/'tle_tailor/helper.py').write_text('VALUE = 2\n')
    elif mutation == 'source-flag-false':
        records[0]['source_unchanged'] = False
    elif mutation == 'runtime-flag-false':
        records[0]['execution_source_unchanged'] = False
    elif mutation == 'no-kernel':
        records[0]['kernel_ready'] = False
    elif mutation == 'failed-worker':
        records[0]['worker_returncode'] = -9
    elif mutation == 'missing-finish':
        del records[0]['finished']
    elif mutation == 'live-options':
        records[0]['validation_options']['catalog'] = True
    else:
        records[0]['jupyter_load_check'] = {}
    assert RELEASE.validate_notebook_execution(records, root)


def test_notebook_manifest_cannot_reduce_the_required_seven(executed_notebook_receipts):
    root, _ = executed_notebook_receipts
    path = root/'tle_tailor/examples/workflows.json'; data = json.loads(path.read_text())
    data['workflows'] = data['workflows'][:1]; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='seven'):
        RELEASE.expected_notebooks(root)


def test_notebook_summary_rejects_nonfinite_values(tmp_path):
    path = tmp_path/'summary.json'; path.write_text('[{"status":"passed","finished":NaN}]')
    with pytest.raises(ValueError, match='Nonfinite'):
        RELEASE.read_notebook_summary(path)


def test_failed_notebook_command_is_not_waived_by_engineering_mode(simulated_gate_commands, monkeypatch, tmp_path):
    args, _ = simulated_gate_commands
    path = tmp_path/'summary.json'; path.write_text('[]')
    args.notebook_execution_report = path
    args.notebook_command_status = 'failure'
    args.engineering_only = True
    # The successful receipt validator is controlled here to isolate the CLI
    # outcome guard: an older valid receipt must not hide a failed new command.
    monkeypatch.setattr(RELEASE, 'validate_notebook_execution', lambda records, root: [])
    report = RELEASE.run_checks(args)
    assert report['exit_status'] == 1 and report['release_ready'] is False
    result = next(item for item in report['checks'] if item['name'] == 'fresh-kernel-notebook-execution')
    assert result['status'] == 'failed' and 'command' in result['detail']


def test_primary_notebook_cannot_disappear_from_declared_receipts(executed_notebook_receipts):
    root, _ = executed_notebook_receipts
    manifest_path = root/'tle_tailor/examples/workflows.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['standalone_companions'] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='deployment companion'):
        RELEASE.expected_notebooks(root)


def test_scoped_release_keeps_whole_checkout_and_diagnostic_limits_visible_as_context(simulated_gate_commands):
    args, _ = simulated_gate_commands
    manifest = args.root/'tle_tailor/examples/workflows.json'
    manifest.write_text(json.dumps({'workflows': [dict(id='reported-diagnostic', status='diagnostic',
        original_differences=['Physical fit and hold-out limits are documented.'])]}))
    report = RELEASE.run_checks(args)
    checks = {item['name']: item for item in report['checks']}
    for name in ('repository-redistribution', 'diagnostic-fit-qualification'):
        assert checks[name]['status'] == 'blocked'
        assert checks[name]['scope'] == 'context' and checks[name]['required'] is False
    assert checks['diagnostic-fit-qualification']['diagnostics'][0]['id'] == 'reported-diagnostic'
    assert checks['legacy-native-engine-accuracy']['required'] is True
    assert report['release_ready'] is False


def test_failed_native_evidence_still_blocks_scoped_release(simulated_gate_commands):
    args, _ = simulated_gate_commands
    receipt = args.root/'native.json'
    receipt.write_text(json.dumps({'status': 'failed', 'release_gate_passed': False,
        'blockers': ['Measured strict full-precision differences remain.']}))
    args.legacy_engine_report = receipt
    args.engineering_only = True
    report = RELEASE.run_checks(args)
    native = next(item for item in report['checks'] if item['name'] == 'legacy-native-engine-accuracy')
    assert native['status'] == 'failed' and native['required'] is True
    assert report['engineering_passed'] is False and report['release_ready'] is False
    assert report['exit_status'] == 1
