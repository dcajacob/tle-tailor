"""Adversarial boundary tests for the standard-library distribution gate."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from audit_distribution import audit, matches


class DistributionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'source'
        self.root.mkdir()
        self.write('tle_tailor/__init__.py', '__version__="0.0rc1"\n')
        self.write('tle_tailor/data/default_config.toml', '[gmat]\napi_path =\n')
        self.write('tle_tailor/validation/public_baseline_sha256.json', '{"revision":"fixture","files":{}}\n')
        for name in ('LICENSE', 'NOTICE', 'licenses/python-sgp4-MIT.txt'):
            self.write(name, 'Synthetic notice fixture; not a real license grant.\n')
        self.write('docs/redistribution.md', 'Synthetic gate fixture.\n')
        self.write('scripts/audit_distribution.py', '# fixture gate\n')
        self.write('tle_tailor/examples/workflows.json', '{"schema_version":1,"workflows":[],"preserved_studies":[]}\n')
        self.policy = {'schema_version': 1, 'scope': 'synthetic gate fixture',
            'runtime_files': ['tle_tailor/__init__.py', 'tle_tailor/data/default_config.toml',
                              'tle_tailor/validation/public_baseline_sha256.json',
                              'tle_tailor/examples/workflows.json'],
            'runtime_globs': ['tle_tailor/examples/notebooks/maintained/**/*.ipynb'],
            'source_extra_files': ['LICENSE', 'NOTICE', 'licenses/python-sgp4-MIT.txt',
                                   'pyproject.toml', 'distribution-policy.json', 'docs/redistribution.md',
                                   'scripts/audit_distribution.py'], 'source_extra_globs': [],
            'exclude_globs': ['**/*.bsp'], 'vendor_source_sha256': {},
            'notice_sha256': {n: self.digest((self.root/n).read_bytes()) for n in ('LICENSE','NOTICE','licenses/python-sgp4-MIT.txt')},
            'reference_manifest': 'tle_tailor/validation/public_baseline_sha256.json',
            'reference_manifest_sha256': self.digest((self.root/'tle_tailor/validation/public_baseline_sha256.json').read_bytes()),
            'repository_publication_status': 'blocked', 'repository_publication_blockers': ['synthetic blocker']}
        self.write('distribution-policy.json', json.dumps(self.policy))
        self.write('pyproject.toml', '''[project]
name="tle-tailor"
version="0.0rc1"
license-files=["LICENSE","NOTICE","licenses/python-sgp4-MIT.txt"]
[tool.uv.build-backend]
source-exclude=["**/*.bsp"]
wheel-exclude=["**/*.bsp"]
''')

    @staticmethod
    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def write(self, name, text):
        target = self.root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def archives(self, wheel_extra=None, omit_notice=False, sdist_extra=None):
        info = 'tle_tailor-0.0rc1.dist-info'
        payload = {n: (self.root/n).read_bytes() for n in self.policy['runtime_files']}
        payload[info+'/METADATA'] = ('Metadata-Version: 2.4\nName: tle-tailor\nVersion: 0.0rc1\n'+
            ''.join('License-File: '+n+'\n' for n in self.policy['notice_sha256'])+'\n').encode()
        payload[info+'/WHEEL'] = b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
        for name in self.policy['notice_sha256']:
            if not omit_notice or name != 'licenses/python-sgp4-MIT.txt':
                payload[info+'/licenses/'+name] = (self.root/name).read_bytes()
        payload.update(wheel_extra or {})
        record = io.StringIO()
        writer = csv.writer(record)
        for name, data in payload.items():
            writer.writerow([name, 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), str(len(data))])
        writer.writerow([info+'/RECORD','',''])
        payload[info+'/RECORD'] = record.getvalue().encode()
        wheel = Path(self.tmp.name)/'fixture.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            for name, data in payload.items():
                archive.writestr(name, data)
        source_payload = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        source_payload.update(sdist_extra or {})
        sdist = Path(self.tmp.name)/'fixture.tar.gz'
        with tarfile.open(sdist, 'w:gz') as archive:
            for name, data in source_payload.items():
                member = tarfile.TarInfo('tle_tailor-0.0rc1/'+name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return wheel, sdist

    def test_clean_boundary_passes_but_repository_stays_blocked(self):
        wheel, sdist = self.archives()
        report = audit(self.root, wheel, sdist)
        self.assertEqual(report['status'], 'passed', report)
        self.assertEqual(report['repository_publication_status'], 'blocked')

    def test_unknown_runtime_source_rejected(self):
        self.write('tle_tailor/unreviewed.py', 'pass\n')
        self.assertEqual(audit(self.root)['status'], 'failed')

    def test_notebook_output_rejected(self):
        self.write('tle_tailor/examples/notebooks/maintained/demo.ipynb', json.dumps({'nbformat':4,'cells':[{'cell_type':'code','outputs':[{'text':'private result'}],'execution_count':1}]}))
        self.assertEqual(audit(self.root)['status'], 'failed')

    def test_notice_mutation_rejected_before_build(self):
        self.write('NOTICE', 'incorrect\n')
        self.assertEqual(audit(self.root)['status'], 'failed')

    def test_frozen_manifest_mutation_rejected(self):
        self.write('tle_tailor/validation/public_baseline_sha256.json', '{"revision":"different","files":{}}\n')
        self.assertEqual(audit(self.root)['status'], 'failed')

    def test_personal_windows_config_path_rejected(self):
        self.write('tle_tailor/data/default_config.toml', '[gmat]\napi_path = C:\\Users\\example\\GMAT\n')
        self.assertEqual(audit(self.root)['status'], 'failed')

    def add_workflow(self, include_companion):
        name = 'tle_tailor/examples/workflows.json'
        script = 'tle_tailor/examples/scripts/demo_script.py'
        self.policy['runtime_files'].append(script)
        self.write(script, '# fixture entrypoint\n')
        self.write('distribution-policy.json', json.dumps(self.policy))
        self.write(name, json.dumps({'workflows':[{'script':script,
            'original_notebook':'tle_tailor/examples/notebooks/history.ipynb',
            'companion':'tle_tailor/examples/notebooks/maintained/demo.ipynb'}]}))
        if include_companion:
            self.write('tle_tailor/examples/notebooks/maintained/demo.ipynb', '{"nbformat":4,"cells":[]}')

    def test_declared_workflow_missing_companion_rejected_before_build(self):
        self.add_workflow(False)
        self.assertEqual(audit(self.root)['status'], 'failed')

    def test_missing_canonical_manifest_cannot_silently_skip_workflow_audit(self):
        name = 'tle_tailor/examples/workflows.json'
        (self.root/name).unlink()
        self.policy['runtime_files'].remove(name)
        self.write('distribution-policy.json', json.dumps(self.policy))
        report = audit(self.root)
        self.assertEqual(report['status'], 'failed')
        check = next(item for item in report['checks'] if item['check'] == 'canonical workflow manifest is present')
        self.assertEqual(check['status'], 'failed')

    def test_scripts_cannot_be_reintroduced_beneath_notebooks(self):
        self.add_workflow(True)
        old = 'tle_tailor/examples/scripts/demo_script.py'
        wrong = 'tle_tailor/examples/notebooks/demo_script.py'
        (self.root/old).rename(self.root/wrong)
        self.policy['runtime_files'].remove(old)
        self.policy['runtime_files'].append(wrong)
        self.write('distribution-policy.json', json.dumps(self.policy))
        path = self.root/'tle_tailor/examples/workflows.json'
        manifest = json.loads(path.read_text())
        manifest['workflows'][0]['script'] = wrong
        path.write_text(json.dumps(manifest))
        report = audit(self.root)
        check = next(item for item in report['checks'] if item['check'] == 'workflow inventory and sanitized companion source set complete')
        self.assertEqual(check['status'], 'failed')
        self.assertIn(wrong, check['invalid_layout'])

    def test_private_script_boundary_survives_split_directory_layout(self):
        private = 'tle_tailor/examples/scripts/local_examples/**'
        self.policy['private_local_globs'] = [private]
        self.write('distribution-policy.json', json.dumps(self.policy))
        project = self.root/'pyproject.toml'
        project.write_text(project.read_text().replace('["**/*.bsp"]', '["**/*.bsp", "'+private+'"]'))
        self.write('tle_tailor/examples/scripts/local_examples/private_script.py', '# local-only fixture\n')
        self.assertEqual(audit(self.root)['status'], 'passed')
        wheel, sdist = self.archives()
        self.assertEqual(audit(self.root, wheel, sdist)['status'], 'failed')

    def test_actual_companion_omitted_from_wheel_rejected(self):
        self.add_workflow(True)
        self.assertEqual(audit(self.root)['status'], 'passed')
        wheel, sdist = self.archives()  # archive fixture intentionally omits glob-only files
        self.assertEqual(audit(self.root, wheel, sdist)['status'], 'failed')

    def test_application_cannot_bypass_import_quarantine_with_reference_exception(self):
        self.policy['quarantined_import_exceptions'] = {'tle_tailor/__init__.py':['import coarse_fit']}
        self.write('distribution-policy.json', json.dumps(self.policy))
        self.write('tle_tailor/__init__.py', 'import coarse_fit\n')
        self.assertEqual(audit(self.root)['status'], 'failed')

    def test_validly_recorded_unknown_wheel_file_rejected(self):
        wheel, sdist = self.archives(wheel_extra={'tle_tailor/unknown.txt':b'unreviewed'})
        self.assertEqual(audit(self.root, wheel, sdist)['status'], 'failed')

    def test_old_example_namespace_cannot_leak_into_new_wheel(self):
        wheel, sdist = self.archives(wheel_extra={'tle_tailor/notebooks/old_script.py':b'# stale copy'})
        report = audit(self.root, wheel, sdist)
        self.assertEqual(report['status'], 'failed')
        check = next(item for item in report['checks'] if item['check'] == 'wheel payload allowlist and source byte identity')
        self.assertIn({'path':'tle_tailor/notebooks/old_script.py',
                       'reason':'undeclared or explicitly excluded payload'}, check['errors'])

    def test_mandatory_gmatpy_dependency_cannot_disappear_from_wheel_metadata(self):
        project = self.root/'pyproject.toml'
        project.write_text(project.read_text().replace('version="0.0rc1"',
                           'version="0.0rc1"\ndependencies=["gmatpy==0.2.1"]'))
        wheel, sdist = self.archives()
        report = audit(self.root, wheel, sdist)
        check = next(item for item in report['checks'] if item['check'] == 'wheel requires pinned GMAT Py without an extra marker')
        self.assertEqual(check['status'], 'failed')

    def test_missing_upstream_wheel_notice_rejected(self):
        wheel, sdist = self.archives(omit_notice=True)
        self.assertEqual(audit(self.root, wheel, sdist)['status'], 'failed')

    def test_private_sdist_dataset_rejected(self):
        wheel, sdist = self.archives(sdist_extra={'tle_tailor/hidden.bsp':b'provider data'})
        self.assertEqual(audit(self.root, wheel, sdist)['status'], 'failed')

    def test_recursive_glob_does_not_accidentally_exclude_companions(self):
        self.assertTrue(matches('tle_tailor/data/spice/de421.bsp', '**/*.bsp'))
        self.assertTrue(matches('tle_tailor/examples/notebooks/basic/demo.ipynb', 'tle_tailor/examples/notebooks/*/*.ipynb'))
        self.assertFalse(matches('tle_tailor/examples/notebooks/maintained/basic/demo.ipynb', 'tle_tailor/examples/notebooks/*/*.ipynb'))


if __name__ == '__main__':
    unittest.main()
