"""CLI dispatch and Space-Track contracts, with no numerical runs or services."""
import importlib
from types import SimpleNamespace

import pytest

from tle_tailor import cli, spacetrack_auth as auth


LINE1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
LINE2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"
TLE = LINE1 + "\n" + LINE2 + "\n"
SENSITIVE = "test-only-password-and-untrusted-response"


def fake_module(monkeypatch, *, result=0):
    calls = []
    def load(name):
        def main(arguments):
            calls.append((name, arguments))
            return result
        return SimpleNamespace(main=main)
    monkeypatch.setattr(cli.importlib, "import_module", load)
    return calls


def test_no_command_shows_first_use_help_without_importing_workflows(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: pytest.fail("Unexpected workflow import"))
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    for command in ("example", "fit-ephemeris", "validate-roundtrip", "spacetrack"):
        assert command in output


@pytest.mark.parametrize("command", [[], ["example"], ["fit-ephemeris"], ["validate-roundtrip"], ["spacetrack"]])
def test_help_is_lazy_and_shows_options(command, monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: pytest.fail("Unexpected workflow import"))
    with pytest.raises(SystemExit) as stopped:
        cli.main([*command, "--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_default_example_delegates_to_osculating_deployment_defaults(monkeypatch):
    calls = fake_module(monkeypatch)
    assert cli.main(["example"]) == 0
    assert calls == [("tle_tailor.examples.scripts.deployment", ["--method", "fd"])]


def test_example_forwards_explicit_provider_elements_and_output(monkeypatch):
    calls = fake_module(monkeypatch)
    assert cli.main(["example", "--epoch", "2024-01-02T03:04:05Z", "--elements",
                     "7000", ".01", "98", "20", "30", "40", "--duration", "3600",
                     "--cadence", "60", "--method", "jax", "--output", "new orbit.tle"]) == 0
    assert calls == [("tle_tailor.examples.scripts.deployment", [
        "--method", "jax", "--epoch", "2024-01-02T03:04:05Z", "--elements",
        "7000.0", "0.01", "98.0", "20.0", "30.0", "40.0", "--duration", "3600.0",
        "--cadence", "60.0", "--output", "new orbit.tle",
    ])]


def test_ephemeris_forwards_explicit_frame_and_filename(monkeypatch):
    calls = fake_module(monkeypatch)
    assert cli.main(["fit-ephemeris", "observations.fth", "--frame", "ITRS",
                     "--method", "jax", "--output", "result.tle"]) == 0
    assert calls == [("tle_tailor.examples.scripts.ephemeris_to_tle", [
        "observations.fth", "--frame", "ITRS", "--method", "jax", "--output", "result.tle",
    ])]


@pytest.mark.parametrize("arguments", [["fit-ephemeris", "obs.fth"],
                                       ["fit-ephemeris", "obs.fth", "--frame", "J2000"],
                                       ["example", "--elements", "1", "2"],
                                       ["example", "--method", "unknown"]])
def test_invalid_dispatch_arguments_fail_before_loading_workflows(arguments, monkeypatch):
    calls = fake_module(monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 2
    assert calls == []


def test_roundtrip_is_explicit_sgp4_validation_control(monkeypatch):
    calls = fake_module(monkeypatch, result=7)
    assert cli.main(["validate-roundtrip", "--method", "jax"]) == 7
    assert calls == [("tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle", [
        "--propagator", "sgp4", "--input-kind", "tle", "--method", "jax",
    ])]


def test_workflow_exception_is_sanitized(monkeypatch, capsys):
    def broken(name):
        raise RuntimeError(SENSITIVE)
    monkeypatch.setattr(cli.importlib, "import_module", broken)
    assert cli.main(["example"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert SENSITIVE not in output.err
    assert "Traceback" not in output.err
    assert "--help" in output.err


class FakeClient:
    def __init__(self, response=TLE, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def gp(self, **query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.response


def test_spacetrack_defaults_to_fresh_prompts_and_prints_only_tle(monkeypatch, capsys):
    client = FakeClient()
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: client)
    monkeypatch.setattr(auth, "get_spacetrack_client", lambda **kw: pytest.fail("Saved credentials read"))
    assert cli.main(["spacetrack", "--norad-id", "25544"]) == 0
    assert client.calls == [dict(norad_cat_id=25544, orderby="EPOCH desc", limit=1, format="tle")]
    output = capsys.readouterr()
    assert output.out == TLE
    assert output.err == ""


def test_noninteractive_explicitly_uses_compatibility_auth_without_prompt(monkeypatch, capsys):
    client = FakeClient()
    calls = []
    def configured(**options):
        calls.append(options)
        return client
    monkeypatch.setattr(auth, "get_spacetrack_client", configured)
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: pytest.fail("Unexpected prompt"))
    assert cli.main(["spacetrack", "--norad-id", "25544", "--non-interactive"]) == 0
    assert calls == [dict(prompt=False)]
    assert capsys.readouterr().out == TLE


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.2", "２５５４４"])
def test_invalid_norad_id_never_authenticates(value, monkeypatch):
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: pytest.fail("Unexpected authentication"))
    with pytest.raises(SystemExit) as stopped:
        cli.main(["spacetrack", "--norad-id", value])
    assert stopped.value.code == 2


@pytest.mark.parametrize("response", [None, [], "", SENSITIVE, "0 " + SENSITIVE + "\n" + TLE,
                                      TLE.replace("5082", "5083"), TLE + SENSITIVE])
def test_bad_response_is_never_echoed(response, monkeypatch, capsys):
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: FakeClient(response=response))
    assert cli.main(["spacetrack", "--norad-id", "25544"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert SENSITIVE not in output.err
    assert LINE1 not in output.err
    assert "no valid two-line TLE" in output.err


def test_response_for_wrong_catalog_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: FakeClient())
    assert cli.main(["spacetrack", "--norad-id", "25545"]) == 1
    assert capsys.readouterr().out == ""


def test_alpha5_catalog_response_matches_requested_numeric_id(monkeypatch, capsys):
    def alpha5(line):
        body = line[:2] + "A0000" + line[7:68]
        return body + str(sum(int(c) if c.isdigit() else int(c == "-") for c in body) % 10)
    response = alpha5(LINE1) + "\n" + alpha5(LINE2) + "\n"
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: FakeClient(response=response))
    assert cli.main(["spacetrack", "--norad-id", "100000"]) == 0
    assert capsys.readouterr().out == response


def test_invalid_alpha5_letter_cannot_be_printed_as_tle(monkeypatch, capsys):
    def malformed(line):
        body = line[:2] + "I0000" + line[7:68]
        return body + str(sum(int(c) if c.isdigit() else int(c == "-") for c in body) % 10)
    response = malformed(LINE1) + "\n" + malformed(LINE2) + "\n"
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: FakeClient(response=response))
    assert cli.main(["spacetrack", "--norad-id", "180000"]) == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("stage", ["authentication", "query"])
def test_service_errors_hide_exception_and_response_details(stage, monkeypatch, capsys):
    def fresh():
        if stage == "authentication":
            raise RuntimeError(SENSITIVE)
        return FakeClient(error=RuntimeError(SENSITIVE))
    monkeypatch.setattr(auth, "prompt_spacetrack_client", fresh)
    assert cli.main(["spacetrack", "--norad-id", "25544"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert SENSITIVE not in output.err
    assert "Traceback" not in output.err


def test_interrupted_request_has_no_traceback(monkeypatch, capsys):
    monkeypatch.setattr(auth, "prompt_spacetrack_client", lambda: FakeClient(error=KeyboardInterrupt()))
    assert cli.main(["spacetrack", "--norad-id", "25544"]) == 130
    assert capsys.readouterr().err == "Cancelled.\n"
