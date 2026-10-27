import configparser
import os
from importlib.resources import files
from pathlib import Path

# Load config
config = configparser.ConfigParser()

_packaged_data = files("tle_tailor.data")
# Caller-owned mission inputs stay outside installed package directories.
_data_override = os.environ.get("TLE_TAILOR_DATA_DIR")
data_path = Path(_data_override).expanduser() if _data_override else _packaged_data
config_path = _packaged_data.joinpath("default_config.toml")
gmat_path = files("tle_tailor.gmat")
config.read(config_path)

gmat_api_path = config['gmat']['api_path']
_output_path = Path(os.environ.get("TLE_TAILOR_OUTPUT_DIR", Path.cwd() / "output")).expanduser()


def __getattr__(name):
    if name == "output_path":
        # Create generated-output storage only when a caller requests it.
        # Installed package data can be read-only and must remain inputs.
        _output_path.mkdir(parents=True, exist_ok=True)
        return _output_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["config", "data_path", "config_path", "gmat_path", "gmat_api_path", "output_path"]
