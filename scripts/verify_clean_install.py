"""Verify an audited wheel in a fresh external environment; run its Python with -I.

This script does not create an environment or install dependencies. Install the
wheel and published dependency first, then invoke this file with that new
interpreter from outside the source checkout. The receipt binds measured bytes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_matches(wheel, site):
    mismatches = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            target = site / name
            if not target.is_file() or target.read_bytes() != archive.read(name):
                mismatches.append(name)
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--dependency-wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import tle_tailor
    import gmatpy
    source = Path(__file__).resolve().parents[1]
    package = Path(tle_tailor.__file__).resolve()
    site = package.parents[1]
    dependency_site = Path(gmatpy.__file__).resolve().parents[1]
    outside = package.is_relative_to(Path(sys.prefix).resolve()) and not package.is_relative_to(source)
    errors = []
    if not outside or sys.prefix == sys.base_prefix:
        errors.append('application is not installed in an external virtual environment')
    if md.version('tle-tailor') != '0.1.0rc1' or md.version('gmatpy') != '0.2.1':
        errors.append('unexpected project/dependency version')
    app_mismatches = payload_matches(args.wheel, site)
    dep_mismatches = payload_matches(args.dependency_wheel, dependency_site)
    if app_mismatches or dep_mismatches:
        errors.append('installed wheel payload differs from measured artifacts')
    cases = []
    bootstrap = '''
import socket, sys, runpy
_original = socket.socket.connect
def offline(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('clean-install verification is offline')
    return _original(self, address)
socket.socket.connect = offline
from astropy.utils import iers
iers.conf.auto_download = False
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name='__main__')
'''
    with tempfile.TemporaryDirectory(prefix='tle-tailor-clean-') as temp:
        env = {k: v for k, v in os.environ.items() if k not in (
            'PYTHONPATH', 'PYTHONHOME', 'GMAT_API_PATH', 'GMAT_PYTHON',
            'SPACETRACK_USER', 'SPACETRACK_PWD', 'TLE_TAILOR_ENV_FILE')}
        env.update(MPLBACKEND='Agg', MPLCONFIGDIR=temp+'/mpl', XDG_CACHE_HOME=temp+'/cache',
                   TLE_TAILOR_OUTPUT_DIR=temp+'/output', TLE_TAILOR_ENV_FILE=temp+'/absent.env',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for method in ('fd', 'jax'):
            workflows = [
                ('synthetic', 'tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle', ['--method', method]),
                ('opm', 'tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_'+method+'_example_script', ['--no-plots']),
            ]
            for kind, module, options in workflows:
                command = [sys.executable, '-I', '-B', '-c', bootstrap, module, *options]
                try:
                    result = subprocess.run(command, cwd=temp, env=env, capture_output=True, text=True, timeout=180)
                    case = {'input': kind, 'method': method, 'returncode': result.returncode,
                            'stdout': result.stdout, 'stderr': result.stderr}
                except subprocess.TimeoutExpired:
                    case = {'input': kind, 'method': method, 'returncode': None, 'error': 'timeout'}
                cases.append(case)
                if case['returncode'] != 0:
                    errors.append(kind+' '+method+' workflow failed')
    report = {'schema_version': 1, 'status': 'failed' if errors else 'passed',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'outside_checkout': outside,
              'python': sys.version, 'platform': platform.platform(), 'prefix': sys.prefix,
              'package': str(package), 'gmatpy': gmatpy.__file__,
              'project_version': md.version('tle-tailor'), 'gmatpy_version': md.version('gmatpy'),
              'wheel_sha256': sha(args.wheel), 'dependency_wheel_sha256': sha(args.dependency_wheel),
              'app_payload_mismatches': app_mismatches, 'dependency_payload_mismatches': dep_mismatches,
              'versions': {name: md.version(name) for name in ('numpy', 'scipy', 'pandas', 'astropy', 'sgp4', 'skyfield', 'jax', 'jaxlib')},
              'cases': cases, 'errors': errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'outside_checkout': outside, 'cases': len(cases), 'errors': errors}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
