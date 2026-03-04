from __future__ import annotations

import argparse
import json
from pathlib import Path

from .forward_architecture import ForwardArchitectureAnalyzer, parse_input_shape_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Hugging Face / PyTorch model source file by tracing nested forward() "
            "calls, operation-level data flow, inferred parameter dimensions, and architecture plots."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the Python source file (for example: modeling_bert.py).",
    )
    parser.add_argument(
        "--entry-class",
        default=None,
        help="Entry model class name. If omitted, an entry class is selected automatically.",
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        metavar="ARG:D1,D2,...",
        help=(
            "Symbolic input shape for forward() arguments. Repeat for multiple args, "
            "for example: --input-shape input_ids:B,S --input-shape attention_mask:B,S"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory for generated report.json and Mermaid plots.",
    )
    parser.add_argument(
        "--max-ops-in-plot",
        type=int,
        default=140,
        help="Maximum number of operations to include in data-flow Mermaid graph.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the final JSON report to stdout.",
    )
    return parser


def summarize(report: dict) -> str:
    entry = report["entry_class"]
    classes = report["classes"]
    forwards = report["forwards"]
    nested_edges = report["nested_forward_graph"]
    op_count = len(forwards.get(entry, {}).get("operations", []))
    return (
        f"Entry class: {entry}\n"
        f"Classes discovered: {len(classes)}\n"
        f"Nested forward edges: {len(nested_edges)}\n"
        f"Operations in {entry}.forward(): {op_count}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")

    input_shapes = parse_input_shape_overrides(args.input_shape)
    analyzer = ForwardArchitectureAnalyzer()
    report = analyzer.analyze_source(
        source_path=source,
        entry_class=args.entry_class,
        input_shapes=input_shapes,
    )

    output_paths = analyzer.write_outputs(
        report=report,
        output_dir=args.output_dir,
        max_ops_in_plot=args.max_ops_in_plot,
    )

    print(summarize(report))
    print("")
    print("Generated files:")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")

    if args.print_json:
        print("")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
