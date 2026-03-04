# model_tools

Static + runtime architecture analyzer for Hugging Face / PyTorch model source files.

## What this tool does

Given a Python model file (for example `modeling_bert.py`), it can analyze:

1. **Nested `forward()` relationships** across classes/submodules
2. **Operation-level flow** inside each `forward()` function
3. **Data-flow edges** between operation outputs and downstream inputs
4. **Parameter dimensions** for common layer constructors (`Linear`, `Embedding`, `Conv*`, `LayerNorm`, etc.)
5. **Symbolic tensor dimensions** (when you provide `--input-shape` hints)
6. **Architecture plots** as Mermaid diagrams
7. **Runtime call path + branch path + real ATen ops** via dummy execution

Static mode output files:

- `report.json` (full machine-readable analysis)
- `nested_forward_architecture.mmd` (nested forward graph)
- `<EntryClass>_dataflow.mmd` (entry class operation/data-flow graph)
- `architecture_report.md` (report with embedded Mermaid)
- `architecture_report.html` (browser visualization with Mermaid + tables)

Runtime mode output files:

- `runtime_report.json` (full runtime trace details)
- `runtime_module_flow.mmd` (actual module call hierarchy/timing)
- `runtime_ops_timeline.mmd` (actual ATen op sequence/timing)
- `runtime_architecture_report.md` (runtime summary with branch table)
- `runtime_architecture_report.html` (browser visualization with Mermaid + tables)
- `architecture_dashboard.html` (quick entry page linking generated HTML reports)

## Quick start

### Static analysis

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

### Runtime tracing (dummy run, real path)

Runtime mode needs a builder function that returns model + dummy inputs.

Example:

```bash
python3 analyze_model_architecture.py \
  --mode runtime \
  --runtime-builder /path/to/runtime_builder.py:build_runtime_case \
  --output-dir analysis_runtime
```

There is a ready-to-copy example builder in:

- `examples/runtime_builder_hf.py`

Run both static + runtime together:

```bash
python3 analyze_model_architecture.py \
  --mode both \
  --source /path/to/modeling_bert.py \
  --entry-class BertModel \
  --input-shape input_ids:B,S \
  --runtime-builder /path/to/runtime_builder.py:build_runtime_case \
  --output-dir analysis_full
```

## Input shape format

`--input-shape ARG:D1,D2,...`

Examples:

- `--input-shape hidden_states:B,S,H`
- `--input-shape attention_mask:B,1,1,S`

Shapes are symbolic; the tool propagates dimensions through common operations where possible.

## Runtime builder contract

`--runtime-builder` must point to a callable in format:

`module_or_file.py:function_name`

The callable can return one of:

1. `{"model": model, "inputs": ..., "kwargs": ..., "source_path": ..., "entry_class": ...}`
2. `(model, inputs)`
3. `(model, args, kwargs)`

`inputs` can be:

- dict (used as `model(**inputs)`)
- tuple/list/single value (used as positional args)

Minimal example (`runtime_builder.py`):

```python
import inspect

from transformers import BertConfig, BertModel
import torch


def build_runtime_case():
    config = BertConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=30522,
    )
    model = BertModel(config).eval()
    batch, seq = 2, 16
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch, seq)),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
    }
    return {
        "model": model,
        "inputs": inputs,
        "source_path": inspect.getsourcefile(model.__class__),
        "entry_class": model.__class__.__name__,
    }
```

Recommended runtime flags:

- `--runtime-seed 1234`
- `--runtime-max-op-nodes 300`
- `--runtime-max-module-nodes 250`
- `--runtime-enable-grad` (only if you need grad-enabled path)

## Notes and limitations

- This is a **static AST-based analyzer** (no model execution required).
- Dynamic Python behavior and runtime-only branches may not be fully resolved.
- Runtime mode requires PyTorch in your environment and a valid builder function.
- Runtime branch tracing is line-based and reports path taken for `if/else` blocks in traced files.