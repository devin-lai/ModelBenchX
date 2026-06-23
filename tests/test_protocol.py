import json

from modelbenchx.workers import _protocol as P


def test_worker_command_plain():
    assert P.worker_command("py", "modelbenchx.workers.x", "/jd") == [
        "py", "-m", "modelbenchx.workers.x", "/jd"]


def test_worker_command_qos_wrap():
    assert P.worker_command("py", "modelbenchx.workers.x", "/jd", qos="utility") == [
        "taskpolicy", "-c", "utility", "py", "-m", "modelbenchx.workers.x", "/jd"]


def test_interpret_success(tmp_path):
    (tmp_path / P.RESULT).write_text(json.dumps({"raw_ms": [1.0]}))
    out = P.interpret(0, "", tmp_path)
    assert out.ok and not out.crashed
    assert out.result == {"raw_ms": [1.0]}


def test_interpret_handled_error(tmp_path):
    (tmp_path / P.ERROR).write_text(json.dumps({"type": "ValueError", "message": "bad shape"}))
    out = P.interpret(1, "traceback...", tmp_path)
    assert not out.ok and not out.crashed
    assert out.error == "ValueError: bad shape"


def test_interpret_native_abort_is_crash(tmp_path):
    # No result/error files, killed by SIGABRT (-6).
    out = P.interpret(-6, "libc++abi: terminating", tmp_path)
    assert not out.ok and out.crashed
    assert "SIGABRT" in out.error
    assert "terminating" in out.error


def test_interpret_abnormal_exit_is_crash(tmp_path):
    out = P.interpret(3, "", tmp_path)
    assert out.crashed and "exit code 3" in out.error


def test_success_requires_result_file(tmp_path):
    # exit 0 but no result.json -> treated as crash (missing contract file).
    out = P.interpret(0, "", tmp_path)
    assert out.crashed


def test_interpret_exit0_with_error_file_is_handled_error(tmp_path):
    # exit 0 but only error.json present: the error file wins, so it is a handled
    # failure (crashed=False), not a native crash. Pins the file precedence.
    (tmp_path / P.ERROR).write_text(json.dumps({"type": "ValueError", "message": "x"}))
    out = P.interpret(0, "", tmp_path)
    assert not out.ok and not out.crashed
    assert out.error == "ValueError: x"


def test_interpret_truncated_result_is_crash_not_raise(tmp_path):
    # A worker killed by a signal mid-write can leave a truncated result.json.
    # interpret() must contain it as one crashed run, never raise (which would
    # abort the whole sweep and defeat subprocess isolation).
    (tmp_path / P.RESULT).write_text('{"raw_ms": [1.0')  # truncated JSON
    out = P.interpret(0, "killed", tmp_path)
    assert out.crashed and not out.ok


def test_interpret_truncated_error_is_crash_not_raise(tmp_path):
    (tmp_path / P.ERROR).write_text('{"type": "Val')  # truncated JSON
    out = P.interpret(1, "boom", tmp_path)
    assert out.crashed and not out.ok


def test_interpret_corrupt_result_falls_through_to_valid_error(tmp_path):
    # exit 0 with a truncated result.json but a valid error.json: the corrupt
    # result is skipped and the handled error wins (not a native crash).
    (tmp_path / P.RESULT).write_text('{"raw_ms": [1.0')  # truncated
    (tmp_path / P.ERROR).write_text(json.dumps({"type": "ValueError", "message": "bad"}))
    out = P.interpret(0, "", tmp_path)
    assert not out.ok and not out.crashed
    assert out.error == "ValueError: bad"


def test_execute_launch_failure_is_contained(tmp_path):
    # A worker that cannot even launch (missing interpreter / taskpolicy) must be
    # recorded as one failed run, not raised out to abort the whole sweep.
    out = P.execute("/nonexistent/python-xyz", "modelbenchx.workers.x", tmp_path, timeout_s=5)
    assert not out.ok and out.crashed
    assert "failed to launch" in out.error
