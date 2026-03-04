from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .forward_architecture import ForwardArchitectureAnalyzer, parse_input_shape_overrides
from .runtime_architecture import RuntimeArchitectureTracer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze model architecture via static source analysis, runtime dummy-run tracing, or both."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["static", "runtime", "both"],
        default="static",
        help="Analysis mode. static=AST analysis, runtime=real execution trace, both=run both.",
    )
    parser.add_argument(
        "--source",
        default=None,
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
        help="Directory for generated reports and Mermaid plots.",
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
        help="Print generated JSON reports to stdout.",
    )

    parser.add_argument(
        "--runtime-builder",
        default=None,
        help=(
            "Builder callable spec to create (model, dummy_inputs), format: module_or_file.py:function. "
            "Required when --mode runtime or --mode both."
        ),
    )
    parser.add_argument(
        "--runtime-seed",
        type=int,
        default=1234,
        help="Random seed used before runtime dummy run.",
    )
    parser.add_argument(
        "--runtime-trace-file",
        action="append",
        default=[],
        metavar="FILE",
        help="Additional Python file paths to include in runtime line/call tracing.",
    )
    parser.add_argument(
        "--runtime-max-op-nodes",
        type=int,
        default=240,
        help="Maximum ATen op nodes to render in runtime operation timeline Mermaid.",
    )
    parser.add_argument(
        "--runtime-max-module-nodes",
        type=int,
        default=220,
        help="Maximum module call nodes to render in runtime module Mermaid.",
    )
    parser.add_argument(
        "--runtime-enable-grad",
        action="store_true",
        help="Enable grad during runtime trace. By default tracing runs under torch.no_grad().",
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


def summarize_runtime(runtime_report: dict) -> str:
    runtime = runtime_report.get("runtime", {})
    op_summary = runtime_report.get("operation_summary", {})
    module_summary = runtime_report.get("module_summary", {})
    return (
        f"Runtime entry class: {runtime_report.get('entry_class')}\n"
        f"Run duration (ms): {runtime.get('duration_ms', 0):.3f}\n"
        f"Total ATen ops: {op_summary.get('total_ops', 0)}\n"
        f"Total module calls: {module_summary.get('total_module_calls', 0)}\n"
        f"Runtime exception: {runtime.get('exception')}"
    )


def write_dashboard_html(
    output_dir: str | Path,
    static_output_paths: dict,
    runtime_output_paths: dict,
    static_report: dict | None,
    runtime_report: dict | None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_path / "architecture_dashboard.html"

    static_html = static_output_paths.get("html_report")
    runtime_html = runtime_output_paths.get("runtime_html_report")
    static_rel = static_html.name if static_html else None
    runtime_rel = runtime_html.name if runtime_html else None

    static_summary = summarize(static_report) if static_report is not None else "Static analysis not generated."
    runtime_summary = summarize_runtime(runtime_report) if runtime_report is not None else "Runtime trace not generated."

    static_link = (
        f'<a href="{html.escape(static_rel)}" target="_blank">Open static HTML report</a>'
        if static_rel
        else "<span class='muted'>Not available</span>"
    )
    runtime_link = (
        f'<a href="{html.escape(runtime_rel)}" target="_blank">Open runtime HTML report</a>'
        if runtime_rel
        else "<span class='muted'>Not available</span>"
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Architecture Dashboard</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); }}
    pre {{ white-space: pre-wrap; background: #0f172a; color: #e2e8f0; border-radius: 10px; padding: 12px; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }}
    a {{ color: #1d4ed8; text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
    .muted {{ color: #64748b; }}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>Architecture Dashboard</h1>
      <p class="muted">Quick entry page for generated HTML visualizations.</p>
    </div>
    <div class="grid">
      <div class="card">
        <h2>Static Analysis</h2>
        <p>{static_link}</p>
        <pre>{html.escape(static_summary)}</pre>
      </div>
      <div class="card">
        <h2>Runtime Trace</h2>
        <p>{runtime_link}</p>
        <pre>{html.escape(runtime_summary)}</pre>
      </div>
    </div>
  </main>
</body>
</html>
"""
    dashboard_path.write_text(html_text, encoding="utf-8")
    return dashboard_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_static = args.mode in {"static", "both"}
    run_runtime = args.mode in {"runtime", "both"}

    if run_static:
        if args.source is None:
            raise ValueError("--source is required for static mode.")
        source = Path(args.source)
        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")

    if run_runtime and not args.runtime_builder:
        raise ValueError("--runtime-builder is required for runtime mode.")

    static_report = None
    static_output_paths = {}
    runtime_report = None
    runtime_output_paths = {}
    runtime_error = None
    dashboard_path = None

    if run_static:
        input_shapes = parse_input_shape_overrides(args.input_shape)
        analyzer = ForwardArchitectureAnalyzer()
        static_report = analyzer.analyze_source(
            source_path=source,
            entry_class=args.entry_class,
            input_shapes=input_shapes,
        )
        static_output_paths = analyzer.write_outputs(
            report=static_report,
            output_dir=args.output_dir,
            max_ops_in_plot=args.max_ops_in_plot,
        )

    if run_runtime:
        try:
            tracer = RuntimeArchitectureTracer()
            runtime_output_paths = tracer.trace_with_builder(
                builder_spec=args.runtime_builder,
                output_dir=args.output_dir,
                seed=args.runtime_seed,
                max_op_nodes=args.runtime_max_op_nodes,
                max_module_nodes=args.runtime_max_module_nodes,
                disable_grad=not args.runtime_enable_grad,
                trace_files=args.runtime_trace_file,
            )
            runtime_report_path = runtime_output_paths["runtime_report_json"]
            runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            runtime_error = str(exc)

    if static_report is not None:
        print("Static analysis summary:")
        print(summarize(static_report))
        print("")

    if runtime_report is not None:
        print("Runtime trace summary:")
        print(summarize_runtime(runtime_report))
        print("")
    elif run_runtime and runtime_error is not None:
        print("Runtime trace summary:")
        print(f"Failed: {runtime_error}")
        print("")

    print("Generated files:")
    for key, path in static_output_paths.items():
        print(f"- {key}: {path}")
    for key, path in runtime_output_paths.items():
        print(f"- {key}: {path}")

    if static_output_paths or runtime_output_paths:
        dashboard_path = write_dashboard_html(
            output_dir=args.output_dir,
            static_output_paths=static_output_paths,
            runtime_output_paths=runtime_output_paths,
            static_report=static_report,
            runtime_report=runtime_report,
        )
        print(f"- architecture_dashboard_html: {dashboard_path}")

    if args.print_json:
        if static_report is not None:
            print("\n--- STATIC REPORT JSON ---\n")
            print(json.dumps(static_report, indent=2))
        if runtime_report is not None:
            print("\n--- RUNTIME REPORT JSON ---\n")
            print(json.dumps(runtime_report, indent=2))
    if run_runtime and runtime_report is None:
        raise SystemExit(f"Runtime tracing failed: {runtime_error}")


if __name__ == "__main__":
    main()
