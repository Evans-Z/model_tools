# model_tools

Static architecture analyzer for Hugging Face / PyTorch model source files.

## What this tool does

Given a Python model file (for example `modeling_bert.py`), it analyzes:

1. **Nested `forward()` relationships** across classes/submodules
2. **Operation-level flow** inside each `forward()` function
3. **Data-flow edges** between operation outputs and downstream inputs
4. **Parameter dimensions** for common layer constructors (`Linear`, `Embedding`, `Conv*`, `LayerNorm`, etc.)
5. **Symbolic tensor dimensions** (when you provide `--input-shape` hints)
6. **Architecture plots** as Mermaid diagrams

Output files:

- `report.json` (full machine-readable analysis)
- `nested_forward_architecture.mmd` (nested forward graph)
- `<EntryClass>_dataflow.mmd` (entry class operation/data-flow graph)
- `architecture_report.md` (report with embedded Mermaid)

## Quick start

Run from repo root:

```bash
python3 analyze_model_architecture.py \
  --source /path/to/modeling_bert.py \
  --entry-class BertModel \
  --input-shape input_ids:B,S \
  --input-shape attention_mask:B,S \
  --output-dir analysis_bert
```

Or use module form:

```bash
python3 -m model_tools.cli --source /path/to/modeling_bert.py
```

## Input shape format

`--input-shape ARG:D1,D2,...`

Examples:

- `--input-shape hidden_states:B,S,H`
- `--input-shape attention_mask:B,1,1,S`

Shapes are symbolic; the tool propagates dimensions through common operations where possible.

## Notes and limitations

- This is a **static AST-based analyzer** (no model execution required).
- Dynamic Python behavior and runtime-only branches may not be fully resolved.
- For exact runtime tensor shapes on all branches, combine this output with runtime tracing.