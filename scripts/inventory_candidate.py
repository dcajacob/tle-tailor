"""Reconcile exact public b9 source against the local package without modifying either."""
import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_REVISION = 'b9c5de8e7cb5ee4a941c6fbbcb24f4d59c3e8c8f'


def inventory(public, candidate):
    records = []
    for old in sorted(public.rglob('*')):
        if not old.is_file():
            continue
        relative = old.relative_to(public).as_posix()
        if relative.startswith('sgp4_jax/'):
            targets = [candidate/'tle_tailor/propagation'/relative]
        elif old.suffix == '.py':
            group = 'core' if old.stem.startswith(('common', 'coord')) else 'fitting'
            targets = [candidate/'tle_tailor'/group/old.name]
        elif old.suffix == '.ipynb':
            targets = list((candidate/'tle_tailor/examples/notebooks').rglob(old.name))
        elif relative.startswith(('ephemeris/', 'tles/')):
            targets = [candidate/'tle_tailor/data'/relative]
        elif relative.startswith('gmat/'):
            targets = [candidate/'tle_tailor'/relative]
        elif relative.startswith('docs/'):
            targets = [candidate/'tle_tailor'/relative, candidate/relative]
        else:
            targets = [candidate/relative]
        targets = [p for p in targets if p.is_file() and '.ipynb_checkpoints' not in p.parts]
        digest = hashlib.sha256(old.read_bytes()).hexdigest()
        records.append({'public_path': relative, 'public_sha256': digest,
                        'candidate_paths': [p.relative_to(candidate).as_posix() for p in targets],
                        'classification': 'no direct counterpart' if not targets else
                        'byte-identical retained/relocated' if any(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in targets) else
                        'modified counterpart; inspect semantic change',
                        'candidate_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in targets]})
    return {'baseline_revision': PUBLIC_REVISION, 'public_files': len(records), 'files': records,
            'limits': 'Byte lineage inventory is not an accuracy or redistribution clearance claim. Public requirements.txt is superseded by pyproject/lock. Preserved notebooks are historical; maintained companions delegate to scripts.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = inventory(args.public_root.resolve(), root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Inventoried {report['public_files']} exact-public files; {args.output}")


if __name__ == '__main__':
    main()
