"""Qualification controls, independent of hosted CI and native execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

PATH = Path(__file__).parents[2]/'scripts/qualify_compatibility.py'
SPEC = importlib.util.spec_from_file_location('qualification_under_test', PATH)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def test_project_override_is_explicit_and_only_no_dependency_install():
    command = qualification.project_install_command('python', 'project.whl', True, True)
    assert command == qualification.pip_prefix('python') + ['install','--no-deps','--no-index','--ignore-requires-python','project.whl']
    with pytest.raises(ValueError, match='not explicitly'):
        qualification.project_install_command('python','project.whl',True,False)
    assert '--ignore-requires-python' not in qualification.project_install_command('python','project.whl',False,True)


def test_external_environment_removes_source_and_credential_discovery(monkeypatch, tmp_path):
    for key in ['PYTHONPATH','PYTHONHOME','SPACETRACK_USER','GMAT_API_PATH','TLE_TAILOR_DATA_DIR','TLE_TAILOR_EPHEMERIS','PIP_IGNORE_REQUIRES_PYTHON','PIP_NO_DEPS','PIP_NO_DEPENDENCIES','PIP_INDEX_URL']:
        monkeypatch.setenv(key,'deliberately-unusable-test-value')
    environment = qualification.environment(tmp_path)
    assert all(name not in environment for name in ['PYTHONPATH','PYTHONHOME','SPACETRACK_USER','GMAT_API_PATH','TLE_TAILOR_DATA_DIR','TLE_TAILOR_EPHEMERIS','PIP_IGNORE_REQUIRES_PYTHON','PIP_NO_DEPS','PIP_NO_DEPENDENCIES','PIP_INDEX_URL'])
    assert environment['PYTHONSAFEPATH']=='1'
    assert environment['PIP_CONFIG_FILE']==qualification.os.devnull
    assert environment['TLE_TAILOR_ENV_FILE']==str(tmp_path/'absent.env')


def test_installed_payload_requires_exact_wheel_bytes(tmp_path):
    wheel=tmp_path/'project.whl'; site=tmp_path/'site';(site/'pkg').mkdir(parents=True)
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('pkg/example.py','original')
        archive.writestr('project.dist-info/RECORD','installer may rewrite this')
    (site/'pkg/example.py').write_text('original')
    assert qualification.payload_mismatches(wheel,site)==[]
    (site/'pkg/example.py').write_text('changed')
    assert qualification.payload_mismatches(wheel,site)==['pkg/example.py']
    (site/'pkg/example.py').unlink()
    assert qualification.payload_mismatches(wheel,site)==['pkg/example.py']


def test_stable_python_rejects_rc_and_free_threaded(monkeypatch):
    version=SimpleNamespace(releaselevel='candidate')
    monkeypatch.setattr(qualification,'sys',SimpleNamespace(implementation=SimpleNamespace(name='cpython'),version_info=version))
    with pytest.raises(ValueError,match='stable'):
        qualification.stable_interpreter()
    version.releaselevel='final'
    monkeypatch.setattr(qualification.sysconfig,'get_config_var',lambda name:1)
    with pytest.raises(ValueError,match='Free-threaded'):
        qualification.stable_interpreter()


def passing_accuracy():
    return {'passed':True,'comparisons':[{'case':c,'reference':r,'method':m,'passed':True}
        for c in ['leo','geo'] for r in ['sgp4','gmatpy'] for m in ['fd','jax']]}


def test_numerical_qualification_requires_every_fd_jax_reference_case():
    qualification.validate_installed_accuracy(passing_accuracy())
    for mutation in ['empty','missing','duplicate','failed_comparison','failed_report']:
        report=passing_accuracy()
        if mutation=='empty':report['comparisons']=[]
        elif mutation=='missing':report['comparisons'].pop()
        elif mutation=='duplicate':report['comparisons'][-1]=report['comparisons'][0]
        elif mutation=='failed_comparison':report['comparisons'][0]['passed']=False
        else:report['passed']=False
        with pytest.raises(ValueError,match='Incomplete or failed'):
            qualification.validate_installed_accuracy(report)


def test_failed_child_is_not_an_opaque_skip():
    with pytest.raises(RuntimeError,match='did not pass'):
        qualification.required_success({'name':'native-runtime','returncode':1})
    with pytest.raises(RuntimeError,match='did not pass'):
        qualification.required_success({'name':'native-runtime','returncode':None})


def test_jaxlib_cannot_masquerade_as_required_jax():
    from email.message import Message
    metadata=Message();metadata['Requires-Dist']='jaxlib>=0.3'
    with pytest.raises(ValueError,match='unconditional default'):
        qualification.base_requirements(metadata)
    metadata['Requires-Dist']='jax>=0.3; extra == "optional"'
    with pytest.raises(ValueError,match='unconditional default'):
        qualification.base_requirements(metadata)
    metadata['Requires-Dist']='jax>=0.3'
    assert qualification.base_requirements(metadata)==['jaxlib>=0.3','jax>=0.3']


@pytest.mark.parametrize('minor',[12,13,14,15])
def test_pure_pyyaml_exception_is_only_python315(monkeypatch,tmp_path,minor):
    monkeypatch.setattr(qualification,'sys',SimpleNamespace(version_info=(3,minor,0)))
    monkeypatch.setenv('PYYAML_FORCE_LIBYAML','1')
    options=qualification.dependency_binary_options()
    environment=qualification.environment(tmp_path)
    assert '--only-binary=:all:' in options
    if minor==15:
        assert options==['--only-binary=:all:','--no-binary=pyyaml']
        assert environment['PYYAML_FORCE_LIBYAML']=='0'
    else:
        assert options==['--only-binary=:all:']
        assert 'PYYAML_FORCE_LIBYAML' not in environment


def test_pure_pyyaml_verification_requires_disabled_extension_and_no_native_payload():
    evidence={'force_libyaml':'0','with_libyaml':False,'root_is_purelib':True,'native_yaml_members':[]}
    qualification.validate_pure_pyyaml(evidence)
    for field,value in [('force_libyaml',None),('force_libyaml','1'),('with_libyaml',True),
                        ('root_is_purelib',False),('native_yaml_members',['yaml/_yaml.so'])]:
        with pytest.raises(ValueError,match='compiler-free pure-Python'):
            qualification.validate_pure_pyyaml(dict(evidence,**{field:value}))


def test_pure_pyyaml_source_receipt_requires_official_sdist_hash():
    item={'metadata':{'name':'PyYAML','version':'6.0.3'},'download_info':{
        'url':'https://files.pythonhosted.org/packages/example/pyyaml-6.0.3.tar.gz',
        'archive_info':{'hashes':{'sha256':'a'*64}}}}
    report={'install':[item]}
    evidence=qualification.pyyaml_source_evidence(report)
    assert evidence['version']=='6.0.3' and evidence['sdist_sha256']=='a'*64
    item['download_info']['url']='https://files.pythonhosted.org/example.whl'
    with pytest.raises(ValueError,match='sdist URL/SHA256'):
        qualification.pyyaml_source_evidence(report)
