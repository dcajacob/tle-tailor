"""Qualify one native interpreter against an exact noneditable built wheel.

Dependency Requires-Python constraints are always respected. A temporary,
explicit override can apply ONLY to the project's own wheel during qualification;
that is evidence gathering, not a supported-version metadata claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import sysconfig
import time
import tomllib
import traceback
import venv
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('Expected exactly one wheel METADATA')
        return BytesParser().parsebytes(archive.read(names[0]))


def payload_mismatches(wheel, site):
    with zipfile.ZipFile(wheel) as archive:
        return [name for name in archive.namelist()
                if not name.endswith(('/', '.dist-info/RECORD'))
                and (not (site/name).is_file() or (site/name).read_bytes() != archive.read(name))]


def environment(output):
    result = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'TLE_TAILOR_DATA_DIR', 'TLE_TAILOR_EPHEMERIS')
              and not k.startswith(('SPACETRACK_', 'GMAT_', 'PIP_', 'PYYAML_'))}
    result.update(PYTHONDONTWRITEBYTECODE='1', PYTHONSAFEPATH='1', PIP_CONFIG_FILE=os.devnull, JAX_PLATFORMS='cpu', MPLBACKEND='Agg',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  TLE_TAILOR_ENV_FILE=str(output/'absent.env'),
                  TLE_TAILOR_OUTPUT_DIR=str(output/'generated'),
                  MPLCONFIGDIR=str(output/'cache/matplotlib'), XDG_CACHE_HOME=str(output/'cache'))
    if needs_pure_pyyaml():
        result['PYYAML_FORCE_LIBYAML'] = '0'
    return result


def stable_interpreter(expected_minor=None):
    if sys.implementation.name != 'cpython' or sys.version_info.releaselevel != 'final':
        raise ValueError('Qualification requires stable standard CPython, not a prerelease interpreter')
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('Free-threaded Python is outside this matrix')
    if expected_minor and '.'.join(map(str, sys.version_info[:2])) != expected_minor:
        raise ValueError('Actual Python minor differs from requested matrix cell')


def offline():
    original = socket.socket.connect
    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Qualification runtime checks are offline')
        return original(self, address)
    socket.socket.connect = connect
    from astropy.utils import iers
    iers.conf.auto_download = False


def runtime_worker(args):
    record = {'status':'failed','python':sys.version,'implementation':sys.implementation.name,
              'releaselevel':sys.version_info.releaselevel,'architecture':platform.machine(),
              'gil_disabled':bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}
    try:
        stable_interpreter(args.expected_python)
        if 'PYTHONPATH' in os.environ:
            raise ValueError('PYTHONPATH must be unset for installed-wheel qualification')
        import tle_tailor
        package = Path(tle_tailor.__file__).resolve()
        if (sys.prefix == sys.base_prefix or not package.is_relative_to(Path(sys.prefix).resolve())
                or package.is_relative_to(args.source_root) or args.source_root in map(Path, sys.path)):
            raise ValueError('Application is not imported from the external virtual environment')
        mismatches = payload_mismatches(args.wheel, package.parents[1])
        if mismatches:
            raise ValueError('Installed application bytes differ from the qualified wheel: '+str(mismatches))
        offline()
        import numpy as np
        import pandas as pd
        import jax
        import jaxlib
        from tle_tailor.cli import main as cli
        from tle_tailor.validation.accuracy import make_reference, read_manifest
        from sgp4.api import Satrec, WGS72
        from sgp4.conveniences import jday_datetime
        products = []
        if cli(['example', '--output', str(args.output.parent/'deployment-fd.tle')]) != 0:
            raise ValueError('Default native propagation / FD deployment workflow failed')
        products.append(args.output.parent/'deployment-fd.tle')
        manifest = read_manifest()
        case = next(c for c in manifest['cases'] if c['id'] == 'leo')
        times, states, _, fit_count, _ = make_reference(case, manifest, 'sgp4')
        table = pd.DataFrame(states[:fit_count].reshape(-1,6), columns=['x','y','z','xdot','ydot','zdot'])
        table.insert(0, 'timestamp', pd.DatetimeIndex(times[:fit_count]))
        ephemeris = args.output.parent/'authored-ephemeris.fth'
        table.to_feather(ephemeris)
        for method in ('fd', 'jax'):
            product = args.output.parent/f'ephemeris-{method}.tle'
            if cli(['fit-ephemeris', str(ephemeris), '--frame', 'TEME', '--method', method,
                    '--output', str(product)]) != 0:
                raise ValueError(method+' supplied-ephemeris workflow failed')
            products.append(product)
        for path in products:
            lines = path.read_text().strip().splitlines()
            if len(lines) != 2 or any(len(line) != 69 for line in lines):
                raise ValueError('Incomplete exported TLE')
            if any(sum(int(c) if c.isdigit() else int(c=='-') for c in line[:-1])%10 != int(line[-1]) for line in lines):
                raise ValueError('Exported TLE checksum invalid')
            model = Satrec.twoline2rv(*lines,WGS72)
            code,r,v = model.sgp4(model.jdsatepoch,model.jdsatepochF)
            if code or not np.isfinite([r,v]).all():
                raise ValueError('Exported TLE cannot be independently propagated')
        if not jax.config.jax_enable_x64 or jax.default_backend() != 'cpu':
            raise ValueError('Native CPU JAX qualification requires enabled float64 and CPU backend')
        if needs_pure_pyyaml():
            import yaml
            distribution = md.distribution('PyYAML')
            wheel = BytesParser().parsebytes((distribution.read_text('WHEEL') or '').encode())
            pure = {'version':distribution.version,
                    'force_libyaml':os.environ.get('PYYAML_FORCE_LIBYAML'),
                    'with_libyaml':getattr(yaml,'__with_libyaml__',None),
                    'root_is_purelib':wheel.get('Root-Is-Purelib','').lower()=='true',
                    'native_yaml_members':[str(path) for path in distribution.files or []
                        if str(path).startswith('yaml/') and str(path).endswith(('.so','.pyd','.dll','.dylib'))]}
            validate_pure_pyyaml(pure)
            record['pyyaml_pure_python'] = pure
        record.update(status='passed', outside_checkout=True, python=sys.version,
                      interpreter=str(Path(sys.executable).absolute()), prefix=sys.prefix,
                      architecture=platform.machine(), platform=platform.platform(), package=str(package),
                      wheel_sha256=sha(args.wheel), app_payload_mismatches=mismatches,
                      source_root_on_python_path=False, pythonpath_unset=True,
                      native_fd_deployment=True, supplied_ephemeris_methods=['fd','jax'],
                      exported_products=len(products), jax_backend=jax.default_backend(), jax_devices=[str(d) for d in jax.devices()],
                      jax_x64=bool(jax.config.jax_enable_x64),
                      versions={name:md.version(name) for name in ('tle-tailor','gmatpy','numpy','scipy','pandas','astropy','sgp4','skyfield','jax','jaxlib')})
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return 0 if record['status'] == 'passed' else 1


def needs_pure_pyyaml():
    return sys.version_info[:2] == (3, 15)


def dependency_binary_options():
    # PyYAML6.0.3 lacks cp315 wheels but officially supports a pure-Python build.
    # All scientific/native dependencies, including gmatpy, remain wheel-only.
    return ['--only-binary=:all:'] + (['--no-binary=pyyaml'] if needs_pure_pyyaml() else [])


def validate_pure_pyyaml(evidence):
    if (evidence.get('force_libyaml') != '0' or evidence.get('with_libyaml') is not False
        or evidence.get('root_is_purelib') is not True or evidence.get('native_yaml_members') != []):
        raise ValueError('Python3.15 PyYAML must use the explicit compiler-free pure-Python path')


def pyyaml_source_evidence(report):
    rows = [item for item in report.get('install', [])
            if item.get('metadata', {}).get('name', '').lower().replace('_','-') == 'pyyaml']
    if len(rows) != 1:
        raise ValueError('Need exactly one PyYAML source-install record')
    row = rows[0]; download = row.get('download_info', {})
    digest = download.get('archive_info', {}).get('hashes', {}).get('sha256', '')
    url = download.get('url', '')
    if (not url.startswith('https://files.pythonhosted.org/') or not url.endswith('.tar.gz')
        or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Missing official PyYAML sdist URL/SHA256 in pip report')
    return {'version':row['metadata']['version'], 'sdist_url':url, 'sdist_sha256':digest,
            'build_classification':'pure-Python wheel assembly, not native compilation'}


def pip_prefix(python):
    return [python, '-m', 'pip', '--isolated', '--disable-pip-version-check', '--keyring-provider', 'disabled']


def base_requirements(metadata):
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        from packaging.requirements import Requirement
    parsed = [(text, Requirement(text)) for text in metadata.get_all('Requires-Dist', [])]
    names = {req.name.lower().replace('_','-') for _,req in parsed if req.marker is None}
    if not {'jax','jaxlib'} <= names:
        raise ValueError('This matrix requires unconditional default JAX and jaxlib dependencies')
    return [text for text,req in parsed if req.marker is None or req.marker.evaluate({'extra':''})]


def source_identity(root):
    names = subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    hashes = {}
    for name in sorted(set(names)-{''}):
        path = root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Source fingerprint needs regular in-checkout files: '+name)
        hashes[name] = sha(path)
    return {'source_fingerprint_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'source_file_count':len(hashes),
            'source_dirty_status':subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).splitlines(),
            'pyproject_sha256':sha(root/'pyproject.toml'),'lock_sha256':sha(root/'uv.lock')}


def project_install_command(python, wheel, needed, allowed):
    if needed and not allowed:
        raise ValueError('Project metadata override was not explicitly authorized')
    command = pip_prefix(python) + ['install', '--no-deps', '--no-index']
    if needed:
        command.append('--ignore-requires-python')
    return command + [wheel]


def validate_installed_accuracy(report):
    expected = {(c,r,m) for c in ('leo','geo') for r in ('sgp4','gmatpy') for m in ('fd','jax')}
    comparisons = report.get('comparisons', [])
    if (report.get('passed') is not True or len(comparisons) != 8
        or {(r.get('case'),r.get('reference'),r.get('method')) for r in comparisons} != expected
        or not all(r.get('passed') is True for r in comparisons)):
        raise ValueError('Incomplete or failed mandatory installed-wheel numerical comparisons')


def run(command, name, output, env, cwd, timeout=1800):
    started = time.monotonic()
    with (output/(name+'.log')).open('w') as log:
        try:
            print(f'BEGIN {name}', flush=True)
            with subprocess.Popen(list(map(str, command)), env=env, cwd=cwd,
                                  stdout=log, stderr=subprocess.STDOUT) as child:
                deadline = started + timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        child.kill(); child.wait()
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        code = child.wait(timeout=min(30., remaining))
                        break
                    except subprocess.TimeoutExpired:
                        print(f'RUNNING {name}: {time.monotonic()-started:.0f}s', flush=True)
            error = None
        except (OSError, subprocess.TimeoutExpired) as exc:
            code, error = None, str(exc)
    print(f'END {name}: exit={code}, {time.monotonic()-started:.1f}s', flush=True)
    return {'name': name, 'command': list(map(str, command)), 'returncode': code,
            'error': error, 'elapsed_seconds': time.monotonic()-started}


def required_success(step):
    if step['returncode'] != 0:
        raise RuntimeError(f"{step['name']} did not pass; see its log")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--sdist', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--expected-python')
    parser.add_argument('--expected-architecture', choices=('x86_64', 'arm64'))
    parser.add_argument('--allow-project-metadata-override', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.wheel = args.wheel.resolve(); args.output = args.output.resolve(); args.source_root = args.source_root.resolve()
    if args.worker:
        return runtime_worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    steps = []
    report = {'schema_version': 1, 'status': 'failed', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'wheel_sha256': sha(args.wheel), 'source_commit': subprocess.check_output(
                  ['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip(),
              'qualification_only': True, 'harness_sha256': sha(__file__), 'steps': steps,
              'dependency_source_exception':{'package':'PyYAML','mode':'pure Python; optional LibYAML disabled'} if needs_pure_pyyaml() else None,
              'actual_python':sys.version, 'actual_implementation':sys.implementation.name,
              'actual_releaselevel':sys.version_info.releaselevel, 'actual_architecture':platform.machine(),
              'actual_gil_disabled':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
              'expected_python':args.expected_python, 'expected_architecture':args.expected_architecture, 'metadata_override': {'requested': args.allow_project_metadata_override}}
    try:
        report.update(source_identity(args.source_root), sdist_sha256=sha(args.sdist) if args.sdist else None)
        stable_interpreter(args.expected_python)
        architecture = platform.machine()
        if args.expected_architecture and architecture != args.expected_architecture:
            raise ValueError(f'Expected native {args.expected_architecture}, got {architecture}')
        if args.output.is_relative_to(args.source_root):
            raise ValueError('Qualification output/environment must be outside the source checkout')
        env = environment(args.output)
        folder = args.output/'venv'
        if folder.exists():
            raise ValueError('Refusing to reuse an existing qualification environment')
        venv.EnvBuilder(with_pip=True).create(folder)
        python = folder/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        # Do not resolve the venv Python symlink into the base interpreter.
        metadata = wheel_metadata(args.wheel)
        requirement = metadata['Requires-Python']
        probe = subprocess.check_output([str(python), '-I', '-c',
            'import sys; from pip._vendor.packaging.specifiers import SpecifierSet; '
            'print(SpecifierSet(sys.argv[1]).contains(".".join(map(str,sys.version_info[:3]))))', requirement], text=True).strip()
        needed = probe != 'True'
        report['metadata_override'].update(needed=needed, used=needed and args.allow_project_metadata_override,
                                           requires_python=requirement, applies_to='project wheel only; dependencies resolved normally')
        if needed and not args.allow_project_metadata_override:
            raise ValueError('Project metadata excludes this interpreter; explicit qualification override required')
        requirements = base_requirements(metadata)
        project = tomllib.loads((args.source_root/'pyproject.toml').read_text())
        requirements += project['dependency-groups']['dev'] + project['dependency-groups']['notebooks']
        req_file = args.output/'requirements.txt'
        req_file.write_text('\n'.join(requirements)+'\n')
        for command, name in [(pip_prefix(python) + ['install', '--upgrade', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', 'pip'], 'pip-bootstrap'),
              (pip_prefix(python) + ['install', *dependency_binary_options(), '--index-url', 'https://pypi.org/simple', '--report', args.output/'dependency-install.json', '-r', req_file], 'dependency-install')]:
            step = run(command, name, args.output, env, args.output); steps.append(step); required_success(step)
        if needs_pure_pyyaml():
            report['pyyaml_source'] = pyyaml_source_evidence(json.loads((args.output/'dependency-install.json').read_text()))
        install = project_install_command(python, args.wheel, needed, args.allow_project_metadata_override)
        step = run(install, 'project-wheel-install', args.output, env, args.output);steps.append(step);required_success(step)
        step = run(pip_prefix(python) + ['check'], 'dependency-check', args.output, env, args.output);steps.append(step);required_success(step)
        step = run([python, '-I', '-B', Path(__file__).resolve(), '--worker', '--wheel', args.wheel,
                    '--output', args.output/'installed-runtime.json', '--source-root', args.source_root,
                    '--expected-python', args.expected_python or '.'.join(map(str,sys.version_info[:2]))],
                   'installed-runtime', args.output, env, args.output, 300);steps.append(step);required_success(step)
        if needs_pure_pyyaml():
            installed = json.loads((args.output/'installed-runtime.json').read_text())['pyyaml_pure_python']
            if installed['version'] != report['pyyaml_source']['version']:
                raise ValueError('Installed PyYAML version differs from pip source receipt')
            report['pyyaml_source']['installed_verification'] = installed
        # These numerical comparisons run from outside checkout, importing the
        # packaged immutable reference and application only from the wheel.
        step = run([python, '-I', '-B', '-m', 'tle_tailor.validation.accuracy', '--cases', 'leo,geo',
                    '--methods', 'fd,jax', '--references', 'sgp4,gmatpy', '--output', args.output/'installed-accuracy'],
                   'installed-accuracy', args.output, env, args.output);steps.append(step);required_success(step)
        accuracy = json.loads((args.output/'installed-accuracy/accuracy-report.json').read_text())
        validate_installed_accuracy(accuracy)
        # Fresh kernels import installed modules; source files are notebook inputs.
        notebooks = run([python, args.source_root/'scripts/validate_notebooks.py', '--workers', '2',
                         '--timeout', '300', '--startup-timeout', '45', '--overall-timeout', '600',
                         '--output', args.output/'notebooks'], 'notebooks', args.output, env, args.output)
        steps.append(notebooks)
        if not args.sdist:
            raise ValueError('Both wheel and sdist are required for full source engineering audit')
        engineering = run([python, args.source_root/'scripts/release_check.py', '--root', args.source_root,
                           '--engineering-only', '--output', args.output/'engineering', '--wheel', args.wheel,
                           '--sdist', args.sdist.resolve(), '--notebook-execution-report', args.output/'notebooks/summary.json',
                           '--notebook-command-status', 'success' if notebooks['returncode'] == 0 else 'failure'],
                          'source-engineering', args.output, env, args.output)
        steps.append(engineering);required_success(notebooks);required_success(engineering)
        report.update(status='passed', actual_python=sys.version, architecture=architecture,
                      installed_runtime=json.loads((args.output/'installed-runtime.json').read_text()),
                      installed_accuracy_comparisons=8, source_engineering_scope='Full existing focused suite, 20 EQN and8 COE public comparisons; source gate explicitly separate from installed-wheel runtime',
                      qualified_without_metadata_override=not needed)
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    (args.output/'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'metadata_override': report['metadata_override'], 'steps': len(steps)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
