"""The production-behavior check must reject test-dependent branches, not ordinary mocks."""
import pytest

from simagent import production_guard as guard


@pytest.mark.parametrize("condition", [
    "getattr(writer, '__module__', None) == 'unittest.mock'",
    "type(writer).__name__ == 'MagicMock'",
    "writer.__module__.startswith('unittest')",
    "isinstance(writer, MagicMock)",
    "isinstance(writer, (io.IOBase, mock.Mock))",
    "'pytest' in sys.modules",
    "os.environ.get('PYTEST_CURRENT_TEST')",
    "hasattr(writer, 'mock_calls')",
])
def test_rejects_test_dependent_condition(condition):
    source = f"if {condition}:\n    alternate_path()\n"
    assert guard.test_runtime_conditions(source, {1, 2})


def test_import_alias_and_multiline_condition():
    source = "from unittest.mock import Mock as Double\nif (\n    isinstance(writer, Double)\n):\n    pass\n"
    assert guard.test_runtime_conditions(source, {3})


def test_preserves_existing_condition_when_only_body_changes():
    source = "if 'pytest' in sys.modules:\n    updated_logging()\n"
    assert guard.test_runtime_conditions(source, {2}) == []


def test_normal_runtime_capability_checks_are_allowed():
    source = "if hasattr(writer, 'fileno'):\n    writer.flush()\nif atomic:\n    rename()\n"
    assert guard.test_runtime_conditions(source, {1, 2, 3, 4}) == []


def test_strings_and_imports_alone_are_not_branches():
    source = "from unittest.mock import Mock\nexample = 'pytest'\n"
    assert guard.test_runtime_conditions(source, {1, 2}) == []


def test_selecting_a_test_runner_is_not_detecting_test_execution():
    source = "if runner == 'pytest':\n    invoke_runner(runner)\n"
    assert guard.test_runtime_conditions(source, {1, 2}) == []


def test_patch_mapping_excludes_tests_and_unchanged_lines():
    patch = ("diff --git a/pkg/io.py b/pkg/io.py\n--- a/pkg/io.py\n+++ b/pkg/io.py\n@@ -10,2 +10,3 @@\n"
             " keep()\n+if isinstance(writer, Mock):\n+    pass\n"
             "diff --git a/tests/test_io.py b/tests/test_io.py\n@@ -0,0 +1,2 @@\n"
             "+if isinstance(writer, Mock):\n+    pass\n")
    assert guard.changed_python_lines(patch) == {"pkg/io.py": {11, 12}}


def test_rejected_submission_can_be_repaired_without_running_submit_command():
    from types import SimpleNamespace
    executed, messages = [], []
    agent = SimpleNamespace(execute_actions=lambda message: executed.append(message),
                            add_messages=lambda *rows: messages.extend(rows),
                            get_template_vars=lambda: {},
                            model=SimpleNamespace(format_observation_messages=lambda message, outputs, variables: outputs))
    problems = [{"file": "pkg/io.py", "line": 4, "condition": "isinstance(writer, Mock)"}]
    guard.install_submit_guard(agent, lambda: problems)
    submit = {"extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/patch.txt"}]}}
    agent.execute_actions(submit)
    assert not executed and messages[0]["returncode"] == 1
    assert len(agent.production_behavior_rejections) == 1
    problems.clear()
    agent.execute_actions(submit)
    assert executed == [submit]


def test_inspection_failure_does_not_block_submission():
    from types import SimpleNamespace
    executed, messages = [], []
    def unavailable():
        raise RuntimeError("source unavailable")
    agent = SimpleNamespace(execute_actions=lambda message: executed.append(message),
                            add_messages=lambda *rows: messages.extend(rows),
                            get_template_vars=lambda: {},
                            model=SimpleNamespace(format_observation_messages=lambda message, outputs, variables: outputs))
    guard.install_submit_guard(agent, unavailable)
    submit = {"extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]}}
    agent.execute_actions(submit)
    assert executed == [submit] and not messages
    assert agent.production_behavior_rejections == []
    assert agent.production_behavior_inspection_errors == ["RuntimeError: source unavailable"]


def test_blocked_submission_observation_renders_with_strict_template():
    from types import SimpleNamespace
    import jinja2
    template = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>{% endif %}"
        "<returncode>{{output.returncode}}</returncode><output>{{output.output}}</output>")
    messages = []
    agent = SimpleNamespace(execute_actions=lambda message: None,
                            add_messages=lambda *rows: messages.extend(rows),
                            get_template_vars=lambda: {},
                            model=SimpleNamespace(format_observation_messages=lambda message, outputs, variables:
                                                  [template.render(output=o) for o in outputs]))
    guard.install_submit_guard(agent, lambda: [{"file": "pkg/io.py", "line": 4, "condition": "isinstance(w, Mock)"}])
    agent.execute_actions({"extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]}})
    assert "<returncode>1</returncode>" in messages[0] and "Submission blocked" in messages[0]


def test_detection_works_without_ast_unparse(monkeypatch):
    import ast
    monkeypatch.delattr(ast, "unparse", raising=False)   # what Python 3.8 looks like
    source = "if isinstance(writer,  Mock):\n    alternate()\n"
    issues = guard.test_runtime_conditions(source, {1})
    assert len(issues) == 1 and issues[0]["condition"] == "isinstance(writer,  Mock)"


def test_repository_inspection_reads_changed_source(tmp_path):
    import subprocess
    source = "if getattr(writer, '__module__', None) == 'unittest.mock':\n    alternate()\n"
    path = tmp_path / "io_helpers.py"
    path.write_text(source)
    patch = ("diff --git a/io_helpers.py b/io_helpers.py\n--- a/io_helpers.py\n+++ b/io_helpers.py\n@@ -0,0 +1,2 @@\n"
             + "".join("+" + line + "\n" for line in source.splitlines()))
    class Env:
        def execute(self, action, timeout=None):
            result = subprocess.run(action["command"], shell=True, capture_output=True, text=True, timeout=timeout)
            return {"output": result.stdout, "returncode": result.returncode}
    issues = guard.inspect_patch(Env(), str(tmp_path), patch)
    assert len(issues) == 1 and issues[0]["file"] == "io_helpers.py"
    assert path.read_text() == source
