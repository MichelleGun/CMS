"""Fixtures for the onboard-camera tests. No drone, no network.

Image helpers live in shapes.py, imported directly by the tests that need them.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: trains a model; takes tens of seconds")


@pytest.fixture
def fresh_telemetry(monkeypatch):
    """Make TelemetryRecorder.snapshot() sample on demand.

    The recorder normally samples on a background thread at sample_hz, which is fine in
    real time but useless under the synthetic clock below: the flight loop would race
    ahead of the sampler and read a pre-takeoff packet forever. Sampling on read keeps
    the telemetry in step with whatever clock the test is using.
    """
    from onboard.kit import TelemetryRecorder
    monkeypatch.setattr(TelemetryRecorder, "snapshot", lambda self: self.sample())


@pytest.fixture
def fast_clock(monkeypatch):
    """Advance fly_hoop's own clock 20 ms per reading and make its sleeps free.

    Only fly_hoop's `time` is patched, not tello_lab.recording's, so telemetry ages stay
    realistically small while the flight loop runs in a fraction of the wall time. Same
    approach as the kit's own fast_survey fixture.
    """
    from onboard import fly_hoop
    clock = [1000.0]

    def monotonic():
        clock[0] += 0.02
        return clock[0]

    monkeypatch.setattr(fly_hoop.time, "monotonic", monotonic)
    monkeypatch.setattr(fly_hoop.time, "sleep", lambda *_a: None)
    return clock


@pytest.fixture
def flight_args():
    """Parsed fly_hoop arguments for a quick dry run, overridable per test."""
    from onboard.fly_hoop import parse

    def make(*extra):
        return parse(["--dry-run", "--countdown", "0", "--settle", "0.2", *extra])
    return make
