import tempfile
import unittest
from pathlib import Path

from model_tools.forward_architecture import ForwardArchitectureAnalyzer


class ForwardArchitectureAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.fixture = self.repo_root / "tests" / "fixtures" / "toy_modeling.py"
        self.analyzer = ForwardArchitectureAnalyzer()

    def test_analyze_toy_model(self) -> None:
        report = self.analyzer.analyze_source(
            source_path=self.fixture,
            entry_class="ToyModel",
            input_shapes={"input_ids": ["B", "S"]},
        )

        self.assertEqual(report["entry_class"], "ToyModel")
        self.assertIn("ToyModel", report["classes"])
        self.assertIn("ToyEncoder", report["forwards"])

        edges = {(edge["from"], edge["to"]) for edge in report["nested_forward_graph"]}
        self.assertIn(("ToyModel", "ToyEncoder"), edges)
        self.assertIn(("ToyEncoder", "ToyLayer"), edges)
        self.assertIn(("ToyLayer", "ToyAttention"), edges)

        emb = report["classes"]["ToyModel"]["submodules"]["embeddings"]
        self.assertEqual(emb["param_shapes"]["weight"], ["config.vocab_size", "config.hidden_size"])

        toy_model_ops = report["forwards"]["ToyModel"]["operations"]
        first_hidden_op = next(op for op in toy_model_ops if "hidden_states" in op["outputs"])
        self.assertEqual(
            first_hidden_op["inferred_output_shapes"]["hidden_states"],
            ["B", "S", "config.hidden_size"],
        )

    def test_write_outputs(self) -> None:
        report = self.analyzer.analyze_source(
            source_path=self.fixture,
            entry_class="ToyModel",
            input_shapes={"input_ids": ["B", "S"]},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.analyzer.write_outputs(report=report, output_dir=tmp_dir, max_ops_in_plot=50)
            for path in paths.values():
                self.assertTrue(path.exists(), f"Expected file to exist: {path}")
            html_report = paths["html_report"].read_text(encoding="utf-8")
            self.assertIn("<html", html_report)
            self.assertIn("mermaid", html_report)


if __name__ == "__main__":
    unittest.main()
