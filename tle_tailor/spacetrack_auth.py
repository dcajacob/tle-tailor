"""Local Space-Track credentials, without secrets in example source or output."""

import argparse
import getpass
import io
import os
import stat
import tempfile
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream
from spacetrack import SpaceTrackClient


class SpaceTrackCredentialsError(RuntimeError):
    """Credentials are missing or cannot be read privately."""


@dataclass(frozen=True, repr=False)
class _Credentials:
    username: str
    password: str


_cached_credentials = None
_cached_client = None
_cached_prompt_file = None
_cached_prompt_base = None


def _project_root(start):
    for directory in (start, *start.parents):
        marker = directory / "pyproject.toml"
        if marker.is_file():
            try:
                project = tomllib.loads(marker.read_text()).get("project", {})
            except (OSError, ValueError):
                continue
            if project.get("name") == "tle-tailor":
                return directory
    return None


def credential_file(env_file=None):
    """Resolve an explicit file, the checkout's .env, or .env in the current directory."""
    specified = env_file or os.environ.get("TLE_TAILOR_ENV_FILE")
    if specified:
        return Path(specified).expanduser().absolute()
    root = _project_root(Path.cwd()) or _project_root(Path(__file__).resolve().parent)
    return (root or Path.cwd()) / ".env"


def _read_credentials(env_file=None):
    username = os.environ.get("SPACETRACK_USER", "")
    password = os.environ.get("SPACETRACK_PWD", "")
    if not (username and password):
        path = credential_file(env_file)
        if path.is_file():
            if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise SpaceTrackCredentialsError(
                    "The credential file is readable by other users. Run chmod 600 on it."
                )
            try:
                # Parse without changing os.environ or expanding password characters.
                values = dotenv_values(path, interpolate=False)
            except OSError:
                raise SpaceTrackCredentialsError("Cannot read the credential file.") from None
            username = username or values.get("SPACETRACK_USER") or ""
            password = password or values.get("SPACETRACK_PWD") or ""
        elif env_file is not None or os.environ.get("TLE_TAILOR_ENV_FILE"):
            raise SpaceTrackCredentialsError("The specified credential file does not exist.")
    return _Credentials(username.strip(), password)


def _hidden_input(label):
    try:
        with warnings.catch_warnings():
            # Never let getpass fall back to visible input on a noninteractive terminal.
            warnings.simplefilter("error", getpass.GetPassWarning)
            return getpass.getpass(label)
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        raise SpaceTrackCredentialsError(
            "Hidden input is unavailable or was cancelled. Run uv run tle-tailor-login "
            "in a terminal, or set SPACETRACK_USER and SPACETRACK_PWD."
        ) from None


def get_spacetrack_client(*, env_file=None, prompt=True):
    """Return a client using environment variables, a private .env, or hidden input.

    Environment variables take precedence. Interactive credentials remain only in
    this Python process; only the setup command writes them to disk. Construction
    does not send a request; Space-Track authenticates when the first query runs.
    """
    global _cached_credentials, _cached_client, _cached_prompt_file, _cached_prompt_base
    credentials = _read_credentials(env_file)
    configured = credentials
    path = credential_file(env_file)
    if (_cached_client is not None and _cached_prompt_file == path
            and _cached_prompt_base == configured):
        return _cached_client
    prompted = False
    if not (credentials.username and credentials.password):
        if not prompt:
            raise SpaceTrackCredentialsError(
                "Space-Track credentials are missing. Run uv run tle-tailor-login."
            )
        username = credentials.username or _hidden_input("Space-Track username (hidden): ").strip()
        password = credentials.password or _hidden_input("Space-Track password: ")
        prompted = True
        credentials = _Credentials(username, password)
        if not (username and password):
            raise SpaceTrackCredentialsError("Space-Track username and password cannot be empty.")
    if credentials != _cached_credentials:
        client = SpaceTrackClient(identity=credentials.username, password=credentials.password)
        _cached_credentials, _cached_client = credentials, client
    _cached_prompt_file = path if prompted else None
    _cached_prompt_base = configured if prompted else None
    return _cached_client


def prompt_spacetrack_client():
    """Create a fresh client from hidden prompts, without discovery or caching.

    This deliberately ignores environment variables, saved credential files,
    and the compatibility helper's process cache. Nothing is persisted.
    Client construction itself does not send a request.
    """
    username = _hidden_input("Space-Track username (hidden): ").strip()
    password = _hidden_input("Space-Track password: ")
    if not (username and password):
        raise SpaceTrackCredentialsError("Space-Track username and password cannot be empty.")
    return SpaceTrackClient(identity=username, password=password)


def clear_spacetrack_session():
    """Discard this helper's cached client and credentials."""
    global _cached_credentials, _cached_client, _cached_prompt_file, _cached_prompt_base
    _cached_credentials = _cached_client = None
    _cached_prompt_file = None
    _cached_prompt_base = None


def _quoted(value):
    if any(character in value for character in ("\n", "\r", "\0")):
        raise SpaceTrackCredentialsError("Credential values must fit on one line.")
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _save_credentials(path, credentials):
    """Replace only credential entries, with an atomic owner-only file write."""
    if path.is_symlink():
        raise SpaceTrackCredentialsError("The credential file must not be a symbolic link.")
    entries = {
        "SPACETRACK_USER": _quoted(credentials.username),
        "SPACETRACK_PWD": _quoted(credentials.password),
    }
    original = path.read_text() if path.exists() else ""
    lines, written = [], set()
    for binding in parse_stream(io.StringIO(original)):
        if binding.key in entries:
            if binding.key not in written:
                lines.append(f"{binding.key}={entries[binding.key]}\n")
                written.add(binding.key)
        else:
            lines.append(binding.original.string)
    if lines and not lines[-1].endswith("\n"):
        lines.append("\n")
    lines.extend(f"{key}={value}\n" for key, value in entries.items() if key not in written)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".env.", delete=False) as stream:
            temporary = Path(stream.name)
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o600)
            stream.write("".join(lines))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Save Space-Track credentials privately in .env.")
    parser.add_argument("--env-file", type=Path, help="Custom credential file path")
    parser.add_argument("--check", action="store_true", help="Check local setup without logging in")
    args = parser.parse_args(argv)
    try:
        if args.check:
            credentials = _read_credentials(args.env_file)
            if not (credentials.username and credentials.password):
                raise SpaceTrackCredentialsError("Credentials are missing. Run uv run tle-tailor-login.")
            print("Space-Track credentials are configured; no network request made.")
            return 0
        credentials = _Credentials(
            _hidden_input("Space-Track username (hidden): ").strip(),
            _hidden_input("Space-Track password: "),
        )
        if not (credentials.username and credentials.password):
            raise SpaceTrackCredentialsError("Space-Track username and password cannot be empty.")
        _save_credentials(credential_file(args.env_file), credentials)
        clear_spacetrack_session()
        print("Space-Track credentials saved in a private .env file. Values were not displayed.")
        return 0
    except (SpaceTrackCredentialsError, OSError) as error:
        message = str(error) if isinstance(error, SpaceTrackCredentialsError) else "Cannot write the credential file."
        print(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
