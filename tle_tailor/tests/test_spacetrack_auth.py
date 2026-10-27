"""Local authentication tests: fake credentials, isolated files, and no network."""

import getpass
import os
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from tle_tailor import spacetrack_auth as auth


FAKE_USER = "fake-user@example.invalid"
FAKE_PASSWORD = "fake-password-for-tests"


@dataclass(repr=False)
class FakeClient:
    identity: str
    password: str

    def login(self):
        raise AssertionError("Authentication tests must never make a network request")


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch, tmp_path):
    """Prevent discovery or opening of any real project credential file."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "tle-tailor"\n')
    monkeypatch.chdir(project)
    monkeypatch.setattr(auth, "__file__", str(tmp_path / "installed" / "spacetrack_auth.py"))
    for key in ("SPACETRACK_USER", "SPACETRACK_PWD", "TLE_TAILOR_ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    original_parser = auth.dotenv_values

    def private_test_parser(path, **options):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return original_parser(path, **options)

    monkeypatch.setattr(auth, "dotenv_values", private_test_parser)
    clients = []

    def make_client(*, identity, password):
        client = FakeClient(identity, password)
        clients.append(client)
        return client

    monkeypatch.setattr(auth, "SpaceTrackClient", make_client)
    auth.clear_spacetrack_session()
    yield project, clients
    auth.clear_spacetrack_session()


def write_private(path, content):
    path.write_text(content)
    path.chmod(0o600)
    return path


def set_prompts(monkeypatch, values):
    values = iter(values)
    labels = []

    def hidden_input(label):
        labels.append(label)
        return next(values)

    monkeypatch.setattr(auth.getpass, "getpass", hidden_input)
    return labels


def assert_no_values(output, *values):
    for value in values:
        assert value not in output


def test_complete_environment_wins_without_reading_file(monkeypatch, isolated_auth):
    project, clients = isolated_auth
    path = project / ".env"
    path.write_text("UNRELATED=unused\n")
    path.chmod(0o644)
    monkeypatch.setenv("SPACETRACK_USER", FAKE_USER)
    monkeypatch.setenv("SPACETRACK_PWD", FAKE_PASSWORD)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Complete environment credentials must not open a file")

    monkeypatch.setattr(auth, "dotenv_values", forbidden_read)
    client = auth.get_spacetrack_client(env_file=path, prompt=False)
    assert client.identity == FAKE_USER
    assert client.password == FAKE_PASSWORD
    assert clients == [client]


@pytest.mark.parametrize("provided", ["username", "password"])
def test_partial_environment_merges_with_private_file_without_exporting(
    monkeypatch, isolated_auth, provided
):
    project, _ = isolated_auth
    write_private(
        project / ".env",
        "SPACETRACK_USER='file-user@example.invalid'\n"
        "SPACETRACK_PWD='file-password-${FAKE_AUTH_TOKEN}'\n",
    )
    monkeypatch.setenv("FAKE_AUTH_TOKEN", "must-not-expand")
    if provided == "username":
        monkeypatch.setenv("SPACETRACK_USER", "  " + FAKE_USER + "  ")
    else:
        monkeypatch.setenv("SPACETRACK_PWD", FAKE_PASSWORD)
    client = auth.get_spacetrack_client(prompt=False)
    assert client.identity == (FAKE_USER if provided == "username" else "file-user@example.invalid")
    assert client.password == (
        "file-password-${FAKE_AUTH_TOKEN}" if provided == "username" else FAKE_PASSWORD
    )
    missing_key = "SPACETRACK_PWD" if provided == "username" else "SPACETRACK_USER"
    assert missing_key not in os.environ


@pytest.mark.parametrize(
    "inner_marker", [None, '[project]\nname = "other-project"\n', "invalid toml = ["]
)
def test_project_file_discovery_from_nested_notebook(
    monkeypatch, isolated_auth, inner_marker
):
    project, _ = isolated_auth
    notebook_directory = project / "tle_tailor" / "examples" / "notebooks" / "gps_examples"
    notebook_directory.mkdir(parents=True)
    if inner_marker is not None:
        (notebook_directory / "pyproject.toml").write_text(inner_marker)
    write_private(
        project / ".env",
        f"SPACETRACK_USER='{FAKE_USER}'\nSPACETRACK_PWD='{FAKE_PASSWORD}'\n",
    )
    monkeypatch.chdir(notebook_directory)
    assert auth.credential_file() == project / ".env"
    assert auth.get_spacetrack_client(prompt=False).identity == FAKE_USER


def test_explicit_file_wins_over_environment_override(monkeypatch, isolated_auth):
    project, _ = isolated_auth
    override = write_private(project / "override.env", "UNRELATED=unused\n")
    explicit = write_private(
        project / "explicit.env",
        f"SPACETRACK_USER='{FAKE_USER}'\nSPACETRACK_PWD='{FAKE_PASSWORD}'\n",
    )
    monkeypatch.setenv("TLE_TAILOR_ENV_FILE", str(override))
    assert auth.credential_file("explicit.env") == explicit
    assert auth.get_spacetrack_client(env_file="explicit.env", prompt=False).identity == FAKE_USER


def test_environment_file_override_is_respected(monkeypatch, isolated_auth):
    project, _ = isolated_auth
    path = write_private(
        project / "private.env",
        f"SPACETRACK_USER='{FAKE_USER}'\nSPACETRACK_PWD='{FAKE_PASSWORD}'\n",
    )
    monkeypatch.setenv("TLE_TAILOR_ENV_FILE", str(path))
    assert auth.credential_file() == path
    assert auth.get_spacetrack_client(prompt=False).password == FAKE_PASSWORD


def test_missing_setup_has_actionable_error_without_prompt_or_client(
    monkeypatch, isolated_auth, capsys
):
    _, clients = isolated_auth
    monkeypatch.setattr(auth.getpass, "getpass", lambda label: pytest.fail("Unexpected prompt"))
    with pytest.raises(auth.SpaceTrackCredentialsError, match="tle-tailor-login"):
        auth.get_spacetrack_client(prompt=False)
    assert clients == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("override", ["argument", "environment"])
def test_missing_explicit_file_is_not_silently_replaced(
    monkeypatch, isolated_auth, override
):
    project, clients = isolated_auth
    missing = project / "absent-private.env"
    options = {"env_file": missing} if override == "argument" else {}
    if override == "environment":
        monkeypatch.setenv("TLE_TAILOR_ENV_FILE", str(missing))
    monkeypatch.setattr(auth.getpass, "getpass", lambda label: pytest.fail("Unexpected prompt"))
    with pytest.raises(auth.SpaceTrackCredentialsError, match="specified credential file"):
        auth.get_spacetrack_client(**options)
    assert clients == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-file permissions")
@pytest.mark.parametrize("mode", [0o644, 0o620, 0o604, 0o601])
def test_permission_guard_rejects_other_user_access(isolated_auth, mode):
    project, clients = isolated_auth
    path = write_private(
        project / ".env",
        f"SPACETRACK_USER='{FAKE_USER}'\nSPACETRACK_PWD='{FAKE_PASSWORD}'\n",
    )
    path.chmod(mode)
    with pytest.raises(auth.SpaceTrackCredentialsError, match="chmod 600"):
        auth.get_spacetrack_client(prompt=False)
    assert clients == []


def test_read_failure_hides_underlying_exception_values(monkeypatch, isolated_auth):
    project, _ = isolated_auth
    write_private(project / ".env", "UNRELATED=unused\n")

    def cannot_read(*args, **kwargs):
        raise OSError(FAKE_PASSWORD)

    monkeypatch.setattr(auth, "dotenv_values", cannot_read)
    with pytest.raises(auth.SpaceTrackCredentialsError, match="Cannot read") as error:
        auth.get_spacetrack_client(prompt=False)
    assert_no_values(str(error.value), FAKE_PASSWORD)
    assert error.value.__suppress_context__


def test_hidden_interactive_credentials_stay_in_process(monkeypatch, isolated_auth, capsys):
    project, clients = isolated_auth
    labels = set_prompts(monkeypatch, ["  " + FAKE_USER + "  ", FAKE_PASSWORD])
    client = auth.get_spacetrack_client()
    assert (client.identity, client.password) == (FAKE_USER, FAKE_PASSWORD)
    assert len(labels) == 2
    assert clients == [client]
    assert not (project / ".env").exists()
    assert "SPACETRACK_USER" not in os.environ
    assert "SPACETRACK_PWD" not in os.environ
    assert capsys.readouterr().out == ""


def test_fresh_prompt_ignores_environment_files_and_cached_session(monkeypatch, isolated_auth, capsys):
    project, clients = isolated_auth
    path = write_private(project / ".env", "UNRELATED=preserved\n")
    before = path.read_bytes()
    monkeypatch.setenv("SPACETRACK_USER", "inherited-user")
    monkeypatch.setenv("SPACETRACK_PWD", "inherited-password")
    cached = FakeClient("cached-user", "cached-password")
    auth._cached_client = cached
    auth._cached_credentials = auth._Credentials(cached.identity, cached.password)
    def forbidden(*a, **kw):
        pytest.fail("Fresh prompts must not discover credentials or storage")
    monkeypatch.setattr(auth, "_read_credentials", forbidden)
    monkeypatch.setattr(auth, "credential_file", forbidden)
    monkeypatch.setattr(auth, "_save_credentials", forbidden)
    labels = set_prompts(monkeypatch, [FAKE_USER, FAKE_PASSWORD])
    client = auth.prompt_spacetrack_client()
    assert (client.identity, client.password) == (FAKE_USER, FAKE_PASSWORD)
    assert clients == [client]
    assert len(labels) == 2
    assert auth._cached_client is cached
    assert path.read_bytes() == before
    assert os.environ["SPACETRACK_USER"] == "inherited-user"
    assert os.environ["SPACETRACK_PWD"] == "inherited-password"
    output = capsys.readouterr()
    assert_no_values(output.out + output.err, FAKE_USER, FAKE_PASSWORD, "cached-password", "inherited-password")


def test_fresh_prompt_does_not_cache_or_reuse_its_client(monkeypatch, isolated_auth):
    _, clients = isolated_auth
    set_prompts(monkeypatch, [FAKE_USER, FAKE_PASSWORD, FAKE_USER, FAKE_PASSWORD])
    first = auth.prompt_spacetrack_client()
    second = auth.prompt_spacetrack_client()
    assert first is not second
    assert clients == [first, second]
    assert auth._cached_client is None
    assert auth._cached_credentials is None


@pytest.mark.parametrize("values", [["", FAKE_PASSWORD], [FAKE_USER, ""]])
def test_fresh_prompt_rejects_empty_credentials(monkeypatch, isolated_auth, values):
    _, clients = isolated_auth
    set_prompts(monkeypatch, values)
    with pytest.raises(auth.SpaceTrackCredentialsError, match="cannot be empty"):
        auth.prompt_spacetrack_client()
    assert clients == []


def test_fresh_prompt_never_falls_back_to_visible_password(monkeypatch, isolated_auth):
    _, clients = isolated_auth
    def unsafe_prompt(label):
        warnings.warn(FAKE_PASSWORD, getpass.GetPassWarning)
        pytest.fail("Visible fallback must not continue")
    monkeypatch.setattr(auth.getpass, "getpass", unsafe_prompt)
    with pytest.raises(auth.SpaceTrackCredentialsError) as stopped:
        auth.prompt_spacetrack_client()
    assert FAKE_PASSWORD not in str(stopped.value)
    assert clients == []
    assert_no_values(repr(auth._Credentials(FAKE_USER, FAKE_PASSWORD)), FAKE_USER, FAKE_PASSWORD)


@pytest.mark.parametrize("provided", ["username", "password"])
@pytest.mark.parametrize("source", ["environment", "file"])
def test_partially_prompted_client_is_cached_for_unchanged_configuration(
    monkeypatch, isolated_auth, provided, source
):
    project, clients = isolated_auth
    key, value = (
        ("SPACETRACK_USER", FAKE_USER)
        if provided == "username" else ("SPACETRACK_PWD", FAKE_PASSWORD)
    )
    if source == "environment":
        monkeypatch.setenv(key, value)
    else:
        write_private(project / ".env", f"{key}='{value}'\n")
    missing_value = FAKE_PASSWORD if provided == "username" else FAKE_USER
    labels = set_prompts(monkeypatch, [missing_value])
    client = auth.get_spacetrack_client()
    assert auth.get_spacetrack_client(prompt=False) is client
    assert len(labels) == 1
    assert clients == [client]


def test_changed_prompt_counterpart_cannot_reuse_cached_client(monkeypatch, isolated_auth):
    _, clients = isolated_auth
    monkeypatch.setenv("SPACETRACK_USER", FAKE_USER)
    labels = set_prompts(monkeypatch, [FAKE_PASSWORD])
    first = auth.get_spacetrack_client()
    monkeypatch.setenv("SPACETRACK_USER", "different-fake-user@example.invalid")
    with pytest.raises(auth.SpaceTrackCredentialsError, match="missing"):
        auth.get_spacetrack_client(prompt=False)
    assert clients == [first]
    assert len(labels) == 1


@pytest.mark.parametrize("source", ["environment", "file"])
def test_removed_configured_credentials_do_not_reuse_stale_client(
    monkeypatch, isolated_auth, source
):
    project, clients = isolated_auth
    if source == "environment":
        monkeypatch.setenv("SPACETRACK_USER", FAKE_USER)
        monkeypatch.setenv("SPACETRACK_PWD", FAKE_PASSWORD)
    else:
        write_private(
            project / ".env",
            f"SPACETRACK_USER='{FAKE_USER}'\nSPACETRACK_PWD='{FAKE_PASSWORD}'\n",
        )
    first = auth.get_spacetrack_client(prompt=False)
    assert auth.get_spacetrack_client(prompt=False) is first
    if source == "environment":
        monkeypatch.delenv("SPACETRACK_USER")
        monkeypatch.delenv("SPACETRACK_PWD")
    else:
        (project / ".env").unlink()
    with pytest.raises(auth.SpaceTrackCredentialsError, match="missing"):
        auth.get_spacetrack_client(prompt=False)
    assert clients == [first]


def test_interactive_cache_respects_file_scope_and_reset(monkeypatch, isolated_auth):
    project, clients = isolated_auth
    labels = set_prompts(monkeypatch, [FAKE_USER, FAKE_PASSWORD, FAKE_USER, FAKE_PASSWORD])
    first = auth.get_spacetrack_client()
    assert auth.get_spacetrack_client(prompt=False) is first
    alternate = write_private(project / "alternate.env", "UNRELATED=unused\n")
    with pytest.raises(auth.SpaceTrackCredentialsError, match="missing"):
        auth.get_spacetrack_client(env_file=alternate, prompt=False)
    auth.clear_spacetrack_session()
    with pytest.raises(auth.SpaceTrackCredentialsError, match="missing"):
        auth.get_spacetrack_client(prompt=False)
    second = auth.get_spacetrack_client()
    assert second is not first
    assert clients == [first, second]
    assert len(labels) == 4


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_cancelled_hidden_prompt_is_safe(monkeypatch, exception):
    def cancelled(label):
        raise exception()

    monkeypatch.setattr(auth.getpass, "getpass", cancelled)
    with pytest.raises(auth.SpaceTrackCredentialsError, match="Hidden input"):
        auth.get_spacetrack_client()


def test_getpass_warning_prevents_visible_fallback(monkeypatch, isolated_auth, capsys):
    _, clients = isolated_auth
    fallback = []

    def unsafe_fallback(label):
        warnings.warn("Synthetic visible-input fallback", getpass.GetPassWarning)
        fallback.append(True)
        return FAKE_PASSWORD

    monkeypatch.setattr(auth.getpass, "getpass", unsafe_fallback)
    assert auth.main([]) == 1
    assert fallback == []
    assert clients == []
    output = capsys.readouterr()
    assert "Hidden input" in output.out
    assert output.err == ""
    assert_no_values(output.out, FAKE_USER, FAKE_PASSWORD)


@pytest.mark.parametrize("values", [("", FAKE_PASSWORD), ("   ", FAKE_PASSWORD), (FAKE_USER, "")])
def test_empty_prompted_credentials_are_rejected(monkeypatch, isolated_auth, values):
    _, clients = isolated_auth
    set_prompts(monkeypatch, values)
    with pytest.raises(auth.SpaceTrackCredentialsError, match="cannot be empty"):
        auth.get_spacetrack_client()
    assert clients == []


def test_setup_roundtrips_special_characters_and_preserves_unrelated_entries(
    monkeypatch, isolated_auth, capsys
):
    project, clients = isolated_auth
    path = project / ".env"
    unrelated = "# retained comment\nAPP_SETTING='keep # exact'\nMULTILINE=\"one\ntwo\"\n"
    tail = "TAIL=without-final-newline"
    write_private(
        path,
        unrelated
        + "SPACETRACK_USER='old-fake-user'\nSPACETRACK_PWD='old-fake-password'\n"
        + "SPACETRACK_PWD='duplicate-fake-password'\nexport SPACETRACK_USER='duplicate-fake-user'\n"
        + tail,
    )
    path.chmod(0o644)
    username = "fake-user+'quoted'\\name@example.invalid"
    password = "fake 'quote' \\ slash # & = ${FAKE_AUTH_TOKEN} \"double\" : tail "
    monkeypatch.setenv("FAKE_AUTH_TOKEN", "must-not-expand")
    set_prompts(monkeypatch, [username, password])
    replacements = []
    original_replace = auth.os.replace

    def observe_atomic_replace(temporary, target):
        assert Path(temporary).parent == path.parent
        if os.name == "posix":
            assert stat.S_IMODE(Path(temporary).stat().st_mode) == 0o600
        assert path.read_text().startswith(unrelated)
        replacements.append((Path(temporary), Path(target)))
        return original_replace(temporary, target)

    monkeypatch.setattr(auth.os, "replace", observe_atomic_replace)
    assert auth.main([]) == 0
    assert len(replacements) == 1
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = path.read_text()
    assert unrelated in saved
    assert tail in saved
    assert saved.count("SPACETRACK_USER=") == 1
    assert saved.count("SPACETRACK_PWD=") == 1
    assert clients == []
    client = auth.get_spacetrack_client(prompt=False)
    assert (client.identity, client.password) == (username, password)
    assert not list(project.glob(".env.*"))
    output = capsys.readouterr()
    assert output.err == ""
    assert_no_values(output.out, username, password)


def test_failed_atomic_replace_keeps_original_and_removes_temporary_file(
    monkeypatch, isolated_auth, capsys
):
    project, _ = isolated_auth
    original = "UNRELATED='must survive'\n"
    path = write_private(project / ".env", original)
    set_prompts(monkeypatch, [FAKE_USER, FAKE_PASSWORD])

    def failed_replace(*args):
        raise OSError(FAKE_PASSWORD)

    monkeypatch.setattr(auth.os, "replace", failed_replace)
    assert auth.main([]) == 1
    assert path.read_text() == original
    assert not list(project.glob(".env.*"))
    output = capsys.readouterr().out
    assert "Cannot write" in output
    assert_no_values(output, FAKE_USER, FAKE_PASSWORD)


def test_setup_rejects_symlink_without_modifying_target(monkeypatch, isolated_auth, capsys):
    project, _ = isolated_auth
    target = write_private(project / "target.env", "UNRELATED='must survive'\n")
    (project / ".env").symlink_to(target)
    set_prompts(monkeypatch, [FAKE_USER, FAKE_PASSWORD])
    assert auth.main([]) == 1
    assert target.read_text() == "UNRELATED='must survive'\n"
    assert "symbolic link" in capsys.readouterr().out


@pytest.mark.parametrize("character", ["\n", "\r", "\0"])
@pytest.mark.parametrize("field", ["username", "password"])
def test_setup_rejects_multiline_values_without_overwriting(
    monkeypatch, isolated_auth, capsys, character, field
):
    project, _ = isolated_auth
    original = "UNRELATED='must survive'\n"
    path = write_private(project / ".env", original)
    username = FAKE_USER + character + "fake-tail" if field == "username" else FAKE_USER
    password = FAKE_PASSWORD + character + "fake-tail" if field == "password" else FAKE_PASSWORD
    set_prompts(monkeypatch, [username, password])
    assert auth.main([]) == 1
    assert path.read_text() == original
    assert not list(project.glob(".env.*"))
    output = capsys.readouterr().out
    assert "one line" in output
    assert_no_values(output, FAKE_USER, FAKE_PASSWORD)


def test_check_is_local_and_does_not_construct_client(isolated_auth, capsys):
    project, clients = isolated_auth
    write_private(
        project / ".env",
        f"SPACETRACK_USER='{FAKE_USER}'\nSPACETRACK_PWD='{FAKE_PASSWORD}'\n",
    )
    assert auth.main(["--check"]) == 0
    assert clients == []
    output = capsys.readouterr().out
    assert "no network request" in output
    assert_no_values(output, FAKE_USER, FAKE_PASSWORD)


def test_check_missing_setup_returns_safe_nonzero_status(isolated_auth, capsys):
    _, clients = isolated_auth
    assert auth.main(["--check"]) == 1
    assert clients == []
    assert "tle-tailor-login" in capsys.readouterr().out
