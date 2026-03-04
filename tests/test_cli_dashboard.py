import tempfile
import unittest
from pathlib import Path

from model_tools.cli import write_dashboard_html


class CliDashboardTests(unittest.TestCase):
    def test_write_dashboard_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = write_dashboard_html(
                output_dir=tmp_dir,
                static_output_paths={},
                runtime_output_paths={},
                static_report=None,
                runtime_report=None,
            )
            self.assertTrue(Path(out).exists())
            content = Path(out).read_text(encoding="utf-8")
            self.assertIn("Architecture Dashboard", content)
            self.assertIn("<html", content)


if __name__ == "__main__":
    unittest.main()
