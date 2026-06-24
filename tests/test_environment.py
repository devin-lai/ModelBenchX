"""Tests for environment.capture(): cross-platform dispatch and tool_version."""

from modelbenchx import environment


def test_capture_has_core_fields_and_tool_version():
    e = environment.capture()
    assert e.os and e.python_version and e.tool_version
    assert isinstance(e.to_dict(), dict)


def test_linux_branch(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    e = environment.capture()
    assert e.os == "Linux"  # no crash off-Darwin; sysctl not required


# ---- power / thermal state (affects latency trust on laptops) ---------------

def test_parse_cpu_speed_limit():
    # Intel macOS prints the clamp '=' separated.
    therm = (
        "Note: No thermal pressure noted\n"
        "Currently delivered:\n"
        "  CPU_Speed_Limit \t= 80\n"
    )
    assert environment._parse_cpu_speed_limit(therm) == 80
    # Apple Silicon prints it whitespace-separated with NO '=' — the primary
    # target. Keying on '=' alone returned None here, silently dropping the
    # throttle caveat on every Apple Silicon host.
    apple = (
        "Note: No thermal warning level has been recorded\n"
        "System-wide thermal status: CPU Power notify\n"
        " CPU_Speed_Limit         65\n"
    )
    assert environment._parse_cpu_speed_limit(apple) == 65
    assert environment._parse_cpu_speed_limit("nothing here") is None


def test_parse_power_source():
    batt = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=...)\t100%; charged\n"
    assert environment._parse_power_source(batt) == "AC Power"
    batt2 = "Now drawing from 'Battery Power'\n"
    assert environment._parse_power_source(batt2) == "Battery Power"
    assert environment._parse_power_source("") is None


def test_parse_low_power_mode():
    assert environment._parse_low_power_mode("lowpowermode         1\n") is True
    assert environment._parse_low_power_mode("lowpowermode         0\n") is False
    assert environment._parse_low_power_mode("hibernatemode 3\n") is None


def test_environment_has_power_thermal_fields():
    e = environment.capture()
    d = e.to_dict()
    for k in ("power_source", "low_power_mode", "cpu_speed_limit"):
        assert k in d  # present (possibly None off-Darwin / when unavailable)
