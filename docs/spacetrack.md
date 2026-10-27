# Optional Space-Track lookup

The deployment and supplied-ephemeris workflows run without Space-Track. To retrieve one latest TLE using your own account:

```bash
uv run tle-tailor spacetrack --norad-id 25544
```

The terminal prompts for your username and a **hidden password**. It makes the requested online lookup and prints the validated two TLE lines. Credentials stay in memory for this command: they are not saved automatically or taken from command arguments, shell history, notebook cells or saved credentials. The default interactive path ignores existing environment/file credentials. Cancellation, unavailable hidden input and service failures produce a short error without credential values.

This is an optional catalog lookup, not orbit determination or confirmation of a prelaunch object. Do not place passwords in commands or notebooks.

## Existing automated configuration

```bash
uv run tle-tailor spacetrack --norad-id 25544 --non-interactive
```

This explicitly uses the existing `SPACETRACK_USER`/`SPACETRACK_PWD` environment or private `.env` configuration without prompting. A secret manager can supply the environment. `TLE_TAILOR_ENV_FILE` or the Python helper's `env_file` selects an existing private file. Missing credentials fail; values and underlying service errors are not printed.

The older `tle-tailor-login` command remains available for users who explicitly want to save a private plaintext `.env`; it is not run by the new command. On POSIX it writes owner-only permissions, preserves unrelated entries and never accepts a password argument. `tle-tailor-login --check` checks local configuration without logging in. Never include credential files when sharing the project.

For a notebook or Python program, `get_spacetrack_client()` retains its established environment/file/hidden-prompt behavior; `prompt=False` disables prompting. `prompt_spacetrack_client()` always requests fresh in-memory credentials. `clear_spacetrack_session()` clears the older helper's in-memory cache. Avoid printing client objects or credential values.
