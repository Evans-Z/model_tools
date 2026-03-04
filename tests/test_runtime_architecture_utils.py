import unittest
from pathlib import Path

from model_tools.runtime_architecture import (
    AtenOpEvent,
    ModuleCallEvent,
    RuntimeArchitectureTracer,
    collect_if_branches_from_source,
    infer_taken_branches,
    load_runtime_case,
)


class RuntimeArchitectureUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.builder_fixture = self.repo_root / "tests" / "fixtures" / "runtime_builder_fixture.py"
        self.branch_fixture = self.repo_root / "tests" / "fixtures" / "branch_source.py"

    def test_load_runtime_case_dict(self) -> None:
        case = load_runtime_case(f"{self.builder_fixture}:build_dict_case")
        self.assertEqual(case["kwargs"]["x"], 1)
        self.assertEqual(case["kwargs"]["y"], 2)
        self.assertEqual(case["metadata"]["name"], "dict_case")
        self.assertEqual(case["entry_class"], "DummyModel")

    def test_load_runtime_case_tuple(self) -> None:
        case = load_runtime_case(f"{self.builder_fixture}:build_tuple_case")
        self.assertEqual(case["args"], ())
        self.assertEqual(case["kwargs"]["x"], 1)

    def test_load_runtime_case_tuple3(self) -> None:
        case = load_runtime_case(f"{self.builder_fixture}:build_tuple3_case")
        self.assertEqual(case["args"], (1, 2))
        self.assertEqual(case["kwargs"]["z"], 3)

    def test_branch_inference(self) -> None:
        branches = collect_if_branches_from_source(self.branch_fixture)
        self.assertEqual(len(branches), 1)
        branch = branches[0]

        body_line = branch["body_lines"][0]
        report = infer_taken_branches(
            branch_specs=branches,
            executed_lines={str(self.branch_fixture.resolve()): [body_line]},
            source_path=self.branch_fixture,
        )
        self.assertEqual(report[0]["taken"], "if")

        else_line = branch["else_lines"][0]
        report_else = infer_taken_branches(
            branch_specs=branches,
            executed_lines={str(self.branch_fixture.resolve()): [else_line]},
            source_path=self.branch_fixture,
        )
        self.assertEqual(report_else[0]["taken"], "else")

    def test_summarizers(self) -> None:
        tracer = RuntimeArchitectureTracer()
        modules = [
            ModuleCallEvent(
                event_id=1,
                parent_event_id=None,
                module_path="<root>",
                module_type="Root",
                depth=0,
                input_tensors=[],
                output_tensors=[],
                start_ns=0,
                end_ns=2_000_000,
            ),
            ModuleCallEvent(
                event_id=2,
                parent_event_id=1,
                module_path="encoder",
                module_type="Encoder",
                depth=1,
                input_tensors=[],
                output_tensors=[],
                start_ns=500_000,
                end_ns=1_500_000,
            ),
        ]
        module_summary = tracer._summarize_modules(modules)
        self.assertEqual(module_summary["total_module_calls"], 2)
        self.assertEqual(module_summary["unique_modules"], 2)

        ops = [
            AtenOpEvent(
                op_id=1,
                module_event_id=2,
                op_name="aten.add",
                input_tensors=[],
                output_tensors=[],
                start_ns=0,
                end_ns=100_000,
            ),
            AtenOpEvent(
                op_id=2,
                module_event_id=2,
                op_name="aten.add",
                input_tensors=[],
                output_tensors=[],
                start_ns=100_000,
                end_ns=200_000,
            ),
        ]
        op_summary = tracer._summarize_ops(ops)
        self.assertEqual(op_summary["total_ops"], 2)
        self.assertEqual(op_summary["top_ops"][0]["op_name"], "aten.add")

    def test_runtime_html_render(self) -> None:
        tracer = RuntimeArchitectureTracer()
        report = {
            "generated_at": "2026-01-01T00:00:00Z",
            "entry_class": "ToyModel",
            "runtime": {"duration_ms": 3.2, "exception": None},
            "operation_summary": {
                "total_ops": 2,
                "top_ops": [{"op_name": "aten.add", "count": 2, "total_ms": 0.2, "avg_ms": 0.1}],
            },
            "module_summary": {
                "total_module_calls": 1,
                "top_modules_by_time": [
                    {"module_path": "encoder", "count": 1, "total_ms": 1.0, "avg_ms": 1.0}
                ],
            },
            "branch_trace": [{"lineno": 10, "test": "x > 0", "taken": "if"}],
        }
        module_mermaid = "flowchart TD\n  a --> b\n"
        ops_mermaid = "flowchart LR\n  o1 --> o2\n"
        html_text = tracer.render_html_summary(report, module_mermaid, ops_mermaid)
        self.assertIn("<html", html_text)
        self.assertIn("Runtime Architecture Trace", html_text)
        self.assertIn("aten.add", html_text)


if __name__ == "__main__":
    unittest.main()
