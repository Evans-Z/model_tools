from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _require_torch() -> Tuple[Any, Any]:
    try:
        import torch  # type: ignore
        from torch.utils._python_dispatch import TorchDispatchMode  # type: ignore

        return torch, TorchDispatchMode
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Runtime tracing requires PyTorch, but it is not available in this environment. "
            "Install torch and re-run with --mode runtime or --mode both."
        ) from exc


def _abspath(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_repr(value: Any, max_len: int = 120) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{value.__class__.__name__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _is_tensor(torch_mod: Any, value: Any) -> bool:
    try:
        return isinstance(value, torch_mod.Tensor)
    except Exception:
        return False


def _shape_list(value: Sequence[Any]) -> List[str]:
    return [str(dim) for dim in value]


def _tensor_metadata(torch_mod: Any, tensor: Any) -> Dict[str, Any]:
    shape = _shape_list(list(tensor.shape))
    return {
        "shape": shape,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def _extract_tensor_metadata(torch_mod: Any, value: Any, max_items: int = 12) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if len(result) >= max_items:
            return
        if _is_tensor(torch_mod, obj):
            result.append(_tensor_metadata(torch_mod, obj))
            return
        if isinstance(obj, dict):
            for item in obj.values():
                walk(item)
                if len(result) >= max_items:
                    return
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
                if len(result) >= max_items:
                    return
            return
        # Hugging Face ModelOutput / dataclass-like
        if hasattr(obj, "to_tuple") and callable(getattr(obj, "to_tuple")):
            try:
                walk(obj.to_tuple())
                return
            except Exception:
                pass
        if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
            for item in vars(obj).values():
                walk(item)
                if len(result) >= max_items:
                    return

    walk(value)
    return result


def _normalize_forward_inputs(case_inputs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if case_inputs is None:
        return (), {}
    if isinstance(case_inputs, dict):
        return (), dict(case_inputs)
    if isinstance(case_inputs, tuple):
        return case_inputs, {}
    if isinstance(case_inputs, list):
        return tuple(case_inputs), {}
    return (case_inputs,), {}


def _load_builder_callable(builder_spec: str) -> Callable[[], Any]:
    if ":" not in builder_spec:
        raise ValueError(
            f"Invalid --runtime-builder '{builder_spec}'. Expected format: module_or_file.py:function_name"
        )
    module_part, func_name = builder_spec.split(":", 1)
    module_part = module_part.strip()
    func_name = func_name.strip()
    if not module_part or not func_name:
        raise ValueError(
            f"Invalid --runtime-builder '{builder_spec}'. Expected format: module_or_file.py:function_name"
        )

    if module_part.endswith(".py") or os.path.sep in module_part:
        file_path = Path(module_part).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Runtime builder file was not found: {file_path}")
        spec = importlib.util.spec_from_file_location(f"runtime_builder_{file_path.stem}", str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import runtime builder module from: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    if not hasattr(module, func_name):
        raise AttributeError(f"Builder function '{func_name}' was not found in '{module_part}'")
    builder = getattr(module, func_name)
    if not callable(builder):
        raise TypeError(f"Runtime builder '{builder_spec}' must be callable")
    return builder


def load_runtime_case(builder_spec: str) -> Dict[str, Any]:
    builder = _load_builder_callable(builder_spec)
    result = builder()

    if isinstance(result, dict):
        if "model" not in result:
            raise ValueError(f"Builder '{builder_spec}' returned dict without required key 'model'")
        case_inputs = result.get("inputs", result.get("input", {}))
        args, kwargs = _normalize_forward_inputs(case_inputs)
        extra_kwargs = result.get("kwargs", {})
        if extra_kwargs:
            if not isinstance(extra_kwargs, dict):
                raise ValueError("Builder dict key 'kwargs' must be a dict if provided")
            kwargs.update(extra_kwargs)
        return {
            "model": result["model"],
            "args": args,
            "kwargs": kwargs,
            "metadata": result.get("metadata", {}),
            "source_path": result.get("source_path"),
            "entry_class": result.get("entry_class"),
            "entry_function": result.get("entry_function", "forward"),
        }

    if isinstance(result, tuple):
        if len(result) == 2:
            model, case_inputs = result
            args, kwargs = _normalize_forward_inputs(case_inputs)
            return {
                "model": model,
                "args": args,
                "kwargs": kwargs,
                "metadata": {},
                "source_path": None,
                "entry_class": None,
                "entry_function": "forward",
            }
        if len(result) == 3:
            model, args_like, kwargs_like = result
            if not isinstance(kwargs_like, dict):
                raise ValueError("If builder returns 3-tuple, third item must be kwargs dict")
            if isinstance(args_like, tuple):
                args = args_like
            elif isinstance(args_like, list):
                args = tuple(args_like)
            else:
                args = (args_like,)
            return {
                "model": model,
                "args": args,
                "kwargs": dict(kwargs_like),
                "metadata": {},
                "source_path": None,
                "entry_class": None,
                "entry_function": "forward",
            }

    raise ValueError(
        "Runtime builder must return one of:\n"
        "1) dict with at least {'model': ..., 'inputs': ...}\n"
        "2) tuple(model, inputs)\n"
        "3) tuple(model, args, kwargs)"
    )


class IfBranchCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def visit_If(self, node: ast.If) -> Any:
        self.records.append(
            {
                "lineno": int(node.lineno),
                "test": ast.unparse(node.test) if hasattr(ast, "unparse") else "<condition>",
                "body_lines": self._collect_line_span(node.body),
                "else_lines": self._collect_line_span(node.orelse),
            }
        )
        self.generic_visit(node)

    @staticmethod
    def _collect_line_span(statements: List[ast.stmt]) -> List[int]:
        lines: List[int] = []
        for stmt in statements:
            start = getattr(stmt, "lineno", None)
            end = getattr(stmt, "end_lineno", None)
            if start is None:
                continue
            if end is None:
                end = start
            lines.extend(range(int(start), int(end) + 1))
        deduped = sorted(set(lines))
        return deduped


def collect_if_branches_from_source(source_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collector = IfBranchCollector()
    collector.visit(tree)
    return collector.records


def infer_taken_branches(
    branch_specs: List[Dict[str, Any]],
    executed_lines: Dict[str, List[int]],
    source_path: str | Path,
) -> List[Dict[str, Any]]:
    source_key = _abspath(source_path)
    executed = set(executed_lines.get(source_key, []))
    result: List[Dict[str, Any]] = []
    for branch in branch_specs:
        body_hit = any(line in executed for line in branch["body_lines"])
        else_hit = any(line in executed for line in branch["else_lines"])
        if body_hit and not else_hit:
            taken = "if"
        elif else_hit and not body_hit:
            taken = "else"
        elif body_hit and else_hit:
            taken = "ambiguous"
        else:
            taken = "not_executed"
        result.append(
            {
                "lineno": branch["lineno"],
                "test": branch["test"],
                "taken": taken,
                "body_line_count": len(branch["body_lines"]),
                "else_line_count": len(branch["else_lines"]),
            }
        )
    return result


@dataclass
class FunctionCallEvent:
    call_id: int
    parent_call_id: Optional[int]
    function: str
    filename: str
    lineno: int
    start_ns: int
    end_ns: Optional[int] = None
    depth: int = 0

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "function": self.function,
            "filename": self.filename,
            "lineno": self.lineno,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
            "depth": self.depth,
        }


@dataclass
class ModuleCallEvent:
    event_id: int
    parent_event_id: Optional[int]
    module_path: str
    module_type: str
    depth: int
    input_tensors: List[Dict[str, Any]]
    start_ns: int
    output_tensors: List[Dict[str, Any]] = field(default_factory=list)
    end_ns: Optional[int] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "module_path": self.module_path,
            "module_type": self.module_type,
            "depth": self.depth,
            "input_tensors": self.input_tensors,
            "output_tensors": self.output_tensors,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AtenOpEvent:
    op_id: int
    module_event_id: Optional[int]
    op_name: str
    input_tensors: List[Dict[str, Any]]
    output_tensors: List[Dict[str, Any]]
    start_ns: int
    end_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "module_event_id": self.module_event_id,
            "op_name": self.op_name,
            "input_tensors": self.input_tensors,
            "output_tensors": self.output_tensors,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
        }


class _PythonExecutionTracer:
    def __init__(self, include_files: Iterable[str]) -> None:
        self.include_files = {_abspath(path) for path in include_files}
        self.executed_lines: Dict[str, set[int]] = {}
        self._calls: List[FunctionCallEvent] = []
        self._frame_to_call_id: Dict[int, int] = {}
        self._stack: List[int] = []
        self._next_call_id = 1
        self._old_trace: Any = None
        self._old_thread_trace: Any = None

    @property
    def calls(self) -> List[FunctionCallEvent]:
        return self._calls

    def _is_included(self, filename: str) -> bool:
        normalized = _abspath(filename)
        return normalized in self.include_files

    def _trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        filename = _abspath(frame.f_code.co_filename)
        frame_id = id(frame)

        if event == "call":
            if self._is_included(filename):
                parent = self._stack[-1] if self._stack else None
                call_id = self._next_call_id
                self._next_call_id += 1
                function_name = frame.f_code.co_name
                qual = f"{Path(filename).name}:{function_name}"
                ev = FunctionCallEvent(
                    call_id=call_id,
                    parent_call_id=parent,
                    function=qual,
                    filename=filename,
                    lineno=int(frame.f_lineno),
                    start_ns=time.perf_counter_ns(),
                    depth=len(self._stack),
                )
                self._calls.append(ev)
                self._frame_to_call_id[frame_id] = call_id
                self._stack.append(call_id)
                self.executed_lines.setdefault(filename, set()).add(int(frame.f_lineno))
            return self._trace

        if frame_id not in self._frame_to_call_id:
            return self._trace

        if event == "line":
            self.executed_lines.setdefault(filename, set()).add(int(frame.f_lineno))
            return self._trace

        if event in {"return", "exception"}:
            call_id = self._frame_to_call_id.pop(frame_id, None)
            if call_id is not None:
                for ev in reversed(self._calls):
                    if ev.call_id == call_id and ev.end_ns is None:
                        ev.end_ns = time.perf_counter_ns()
                        break
                if self._stack and self._stack[-1] == call_id:
                    self._stack.pop()
                elif call_id in self._stack:
                    self._stack.remove(call_id)
            return self._trace

        return self._trace

    def __enter__(self) -> "_PythonExecutionTracer":
        self._old_trace = sys.gettrace()
        self._old_thread_trace = threading.gettrace()
        sys.settrace(self._trace)
        threading.settrace(self._trace)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        sys.settrace(self._old_trace)
        threading.settrace(self._old_thread_trace)
        now = time.perf_counter_ns()
        for ev in self._calls:
            if ev.end_ns is None:
                ev.end_ns = now

    def to_serializable_lines(self) -> Dict[str, List[int]]:
        return {path: sorted(lines) for path, lines in self.executed_lines.items()}


class _RuntimeState:
    def __init__(self) -> None:
        self.module_stack: List[int] = []


class RuntimeArchitectureTracer:
    def __init__(self) -> None:
        self._torch: Any = None
        self._TorchDispatchMode: Any = None

    def trace_with_builder(
        self,
        builder_spec: str,
        output_dir: str | Path,
        *,
        seed: int = 1234,
        max_op_nodes: int = 240,
        max_module_nodes: int = 220,
        disable_grad: bool = True,
        trace_files: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        case = load_runtime_case(builder_spec)
        report = self.trace_case(
            model=case["model"],
            args=case["args"],
            kwargs=case["kwargs"],
            source_path=case["source_path"],
            entry_class=case["entry_class"],
            entry_function=case["entry_function"],
            metadata=case["metadata"],
            seed=seed,
            disable_grad=disable_grad,
            trace_files=trace_files,
        )
        return self.write_outputs(
            report=report,
            output_dir=output_dir,
            max_op_nodes=max_op_nodes,
            max_module_nodes=max_module_nodes,
        )

    def trace_case(
        self,
        *,
        model: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        source_path: Optional[str] = None,
        entry_class: Optional[str] = None,
        entry_function: str = "forward",
        metadata: Optional[Dict[str, Any]] = None,
        seed: int = 1234,
        disable_grad: bool = True,
        trace_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        torch, TorchDispatchMode = _require_torch()
        self._torch = torch
        self._TorchDispatchMode = TorchDispatchMode

        if metadata is None:
            metadata = {}

        if source_path is None:
            source_path = inspect.getsourcefile(model.__class__)
        if source_path is None:
            raise ValueError(
                "Unable to infer model source file. Provide source_path via runtime builder result."
            )
        source_path = _abspath(source_path)

        if entry_class is None:
            entry_class = model.__class__.__name__

        include_files = [source_path]
        if trace_files:
            include_files.extend(_abspath(path) for path in trace_files)

        branch_specs = collect_if_branches_from_source(source_path)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        module_events: List[ModuleCallEvent] = []
        aten_ops: List[AtenOpEvent] = []
        runtime_state = _RuntimeState()
        next_module_event_id = 1
        next_op_id = 1
        module_handles: List[Any] = []

        module_names = {id(module_obj): name or "<root>" for name, module_obj in model.named_modules()}
        module_events_by_id: Dict[int, ModuleCallEvent] = {}

        def pre_hook(module_obj: Any, input_value: Any) -> None:
            nonlocal next_module_event_id
            parent_id = runtime_state.module_stack[-1] if runtime_state.module_stack else None
            event_id = next_module_event_id
            next_module_event_id += 1
            module_path = module_names.get(id(module_obj), module_obj.__class__.__name__)
            ev = ModuleCallEvent(
                event_id=event_id,
                parent_event_id=parent_id,
                module_path=module_path,
                module_type=module_obj.__class__.__name__,
                depth=len(runtime_state.module_stack),
                input_tensors=_extract_tensor_metadata(torch, input_value),
                start_ns=time.perf_counter_ns(),
            )
            module_events.append(ev)
            module_events_by_id[event_id] = ev
            runtime_state.module_stack.append(event_id)

        def post_hook(module_obj: Any, input_value: Any, output_value: Any) -> None:
            now = time.perf_counter_ns()
            if not runtime_state.module_stack:
                return
            event_id = runtime_state.module_stack.pop()
            ev = module_events_by_id.get(event_id)
            if ev is None:
                return
            ev.output_tensors = _extract_tensor_metadata(torch, output_value)
            ev.end_ns = now

        for module_obj in model.modules():
            module_handles.append(module_obj.register_forward_pre_hook(pre_hook))
            module_handles.append(module_obj.register_forward_hook(post_hook))

        tracer = _PythonExecutionTracer(include_files=include_files)

        class AtenRecorderMode(TorchDispatchMode):  # type: ignore[misc, valid-type]
            def __torch_dispatch__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Optional[Dict[str, Any]] = None) -> Any:
                nonlocal next_op_id
                if kwargs is None:
                    kwargs = {}
                start_ns = time.perf_counter_ns()
                output = func(*args, **kwargs)
                end_ns = time.perf_counter_ns()

                op_name = str(getattr(func, "__name__", getattr(func, "_schema", func)))
                # Try to include namespace and overload packet if available.
                if hasattr(func, "_overloadpacket"):
                    op_name = str(func._overloadpacket)

                module_event_id = runtime_state.module_stack[-1] if runtime_state.module_stack else None
                op_event = AtenOpEvent(
                    op_id=next_op_id,
                    module_event_id=module_event_id,
                    op_name=op_name,
                    input_tensors=_extract_tensor_metadata(torch, (args, kwargs)),
                    output_tensors=_extract_tensor_metadata(torch, output),
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                next_op_id += 1
                aten_ops.append(op_event)
                return output

        run_start_ns = time.perf_counter_ns()
        run_exception: Optional[str] = None
        result_metadata: Dict[str, Any] = {}

        try:
            model.eval()
            with tracer:
                with AtenRecorderMode():
                    if disable_grad:
                        with torch.no_grad():
                            output = model(*args, **kwargs)
                    else:
                        output = model(*args, **kwargs)
            result_metadata["output_tensors"] = _extract_tensor_metadata(torch, output)
            result_metadata["output_repr"] = _safe_repr(output)
        except Exception as exc:
            run_exception = f"{exc.__class__.__name__}: {exc}"
            result_metadata["exception"] = run_exception
        finally:
            run_end_ns = time.perf_counter_ns()
            for handle in module_handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            # Ensure all unclosed module calls are terminated.
            for event in module_events:
                if event.end_ns is None:
                    event.end_ns = run_end_ns

        executed_lines = tracer.to_serializable_lines()
        branch_trace = infer_taken_branches(branch_specs, executed_lines, source_path=source_path)

        module_summary = self._summarize_modules(module_events)
        op_summary = self._summarize_ops(aten_ops)
        call_flow = [call.to_dict() for call in tracer.calls]

        report: Dict[str, Any] = {
            "generated_at": _now_utc_iso(),
            "analysis_mode": "runtime",
            "entry_class": entry_class,
            "entry_function": entry_function,
            "source_path": source_path,
            "builder_metadata": metadata,
            "runtime_config": {
                "seed": seed,
                "disable_grad": disable_grad,
                "trace_files": include_files,
            },
            "runtime": {
                "start_ns": run_start_ns,
                "end_ns": run_end_ns,
                "duration_ms": (run_end_ns - run_start_ns) / 1_000_000,
                "exception": run_exception,
                "input_args_count": len(args),
                "input_kwargs": sorted(kwargs.keys()),
            },
            "model_calls": [event.to_dict() for event in module_events],
            "aten_operations": [event.to_dict() for event in aten_ops],
            "python_function_calls": call_flow,
            "executed_lines": executed_lines,
            "branch_trace": branch_trace,
            "module_summary": module_summary,
            "operation_summary": op_summary,
            "result_metadata": result_metadata,
        }
        return report

    def write_outputs(
        self,
        report: Dict[str, Any],
        output_dir: str | Path,
        *,
        max_op_nodes: int = 240,
        max_module_nodes: int = 220,
    ) -> Dict[str, Path]:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "runtime_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        module_mermaid = self.render_module_flow_mermaid(report, max_nodes=max_module_nodes)
        module_path = output_path / "runtime_module_flow.mmd"
        module_path.write_text(module_mermaid, encoding="utf-8")

        ops_mermaid = self.render_ops_timeline_mermaid(report, max_nodes=max_op_nodes)
        ops_path = output_path / "runtime_ops_timeline.mmd"
        ops_path.write_text(ops_mermaid, encoding="utf-8")

        markdown = self.render_markdown_summary(report, module_mermaid, ops_mermaid)
        md_path = output_path / "runtime_architecture_report.md"
        md_path.write_text(markdown, encoding="utf-8")

        return {
            "runtime_report_json": report_path,
            "runtime_module_mermaid": module_path,
            "runtime_ops_mermaid": ops_path,
            "runtime_markdown_report": md_path,
        }

    def render_module_flow_mermaid(self, report: Dict[str, Any], max_nodes: int = 220) -> str:
        calls = report.get("model_calls", [])
        if not calls:
            return "flowchart TD\n  empty[No module calls captured]\n"

        # Keep earliest max_nodes calls for readability.
        selected_calls = calls[:max_nodes]
        ids = {call["event_id"] for call in selected_calls}
        lines = ["flowchart TD"]
        lines.append("  classDef module fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e;")

        for call in selected_calls:
            node_id = f"m_{call['event_id']}"
            dur = call.get("duration_ms")
            dur_text = f"{dur:.3f} ms" if isinstance(dur, (int, float)) else "?"
            label = (
                f"{call['module_path']}\\n"
                f"{call['module_type']}\\n"
                f"t={dur_text}"
            ).replace('"', "'")
            lines.append(f'  {node_id}["{label}"]:::module')

        for call in selected_calls:
            parent = call.get("parent_event_id")
            if parent is None or parent not in ids:
                continue
            src = f"m_{parent}"
            dst = f"m_{call['event_id']}"
            lines.append(f"  {src} --> {dst}")

        if len(calls) > len(selected_calls):
            note = f"Truncated to first {len(selected_calls)} module calls out of {len(calls)}."
            lines.append(f'  note["{note}"]')

        return "\n".join(lines) + "\n"

    def render_ops_timeline_mermaid(self, report: Dict[str, Any], max_nodes: int = 240) -> str:
        ops = report.get("aten_operations", [])
        if not ops:
            return "flowchart LR\n  empty[No ATen operations captured]\n"

        selected_ops = ops[:max_nodes]
        lines = ["flowchart LR"]
        lines.append("  classDef op fill:#ede9fe,stroke:#6d28d9,color:#4c1d95;")

        for op in selected_ops:
            node_id = f"op_{op['op_id']}"
            outputs = op.get("output_tensors", [])
            first_shape = outputs[0]["shape"] if outputs else []
            shape_text = "(" + ", ".join(first_shape) + ")" if first_shape else "?"
            dur = op.get("duration_ms", 0.0)
            label = f"{op['op_name']}\\n{shape_text}\\n{dur:.3f} ms".replace('"', "'")
            lines.append(f'  {node_id}["{label}"]:::op')

        for i in range(len(selected_ops) - 1):
            src = f"op_{selected_ops[i]['op_id']}"
            dst = f"op_{selected_ops[i + 1]['op_id']}"
            lines.append(f"  {src} --> {dst}")

        if len(ops) > len(selected_ops):
            note = f"Truncated to first {len(selected_ops)} ops out of {len(ops)}."
            lines.append(f'  note["{note}"]')

        return "\n".join(lines) + "\n"

    def render_markdown_summary(
        self,
        report: Dict[str, Any],
        module_mermaid: str,
        ops_mermaid: str,
    ) -> str:
        runtime = report.get("runtime", {})
        module_summary = report.get("module_summary", {})
        op_summary = report.get("operation_summary", {})
        branch_trace = report.get("branch_trace", [])

        top_ops = op_summary.get("top_ops", [])[:12]
        top_modules = module_summary.get("top_modules_by_time", [])[:12]
        branch_rows = branch_trace[:30]

        lines = [
            "# Runtime Architecture Trace",
            "",
            f"- Generated: `{report.get('generated_at')}`",
            f"- Entry class: `{report.get('entry_class')}`",
            f"- Source: `{report.get('source_path')}`",
            f"- Run duration: `{runtime.get('duration_ms', 0):.3f} ms`",
            f"- Exception: `{runtime.get('exception')}`",
            "",
            "## Real module call flow",
            "",
            "```mermaid",
            module_mermaid.rstrip(),
            "```",
            "",
            "## Real ATen operation timeline",
            "",
            "```mermaid",
            ops_mermaid.rstrip(),
            "```",
            "",
            "## Top operations (by count)",
            "",
            "| op | count | total_ms | avg_ms |",
            "|---|---:|---:|---:|",
        ]
        for row in top_ops:
            lines.append(
                f"| `{row['op_name']}` | {row['count']} | {row['total_ms']:.3f} | {row['avg_ms']:.3f} |"
            )
        if not top_ops:
            lines.append("| _none_ | 0 | 0.000 | 0.000 |")

        lines.extend(
            [
                "",
                "## Top modules (by total time)",
                "",
                "| module | calls | total_ms | avg_ms |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in top_modules:
            lines.append(
                f"| `{row['module_path']}` | {row['count']} | {row['total_ms']:.3f} | {row['avg_ms']:.3f} |"
            )
        if not top_modules:
            lines.append("| _none_ | 0 | 0.000 | 0.000 |")

        lines.extend(
            [
                "",
                "## Branch trace (executed path)",
                "",
                "| line | condition | taken |",
                "|---:|---|---|",
            ]
        )
        for row in branch_rows:
            test = str(row["test"]).replace("|", "\\|")
            lines.append(f"| {row['lineno']} | `{test}` | `{row['taken']}` |")
        if not branch_rows:
            lines.append("|  | _no if-branches found_ |  |")

        lines.extend(
            [
                "",
                "Detailed trace data is in `runtime_report.json`.",
                "",
            ]
        )
        return "\n".join(lines)

    def _summarize_modules(self, module_events: List[ModuleCallEvent]) -> Dict[str, Any]:
        by_path: Dict[str, Dict[str, Any]] = {}
        hierarchy_edges: Dict[Tuple[str, str], int] = {}
        id_to_event = {ev.event_id: ev for ev in module_events}

        for ev in module_events:
            duration_ms = ev.duration_ms or 0.0
            slot = by_path.setdefault(
                ev.module_path,
                {"module_path": ev.module_path, "module_type": ev.module_type, "count": 0, "total_ms": 0.0},
            )
            slot["count"] += 1
            slot["total_ms"] += duration_ms

            if ev.parent_event_id is not None and ev.parent_event_id in id_to_event:
                parent = id_to_event[ev.parent_event_id]
                key = (parent.module_path, ev.module_path)
                hierarchy_edges[key] = hierarchy_edges.get(key, 0) + 1

        top = sorted(by_path.values(), key=lambda row: row["total_ms"], reverse=True)
        for row in top:
            row["avg_ms"] = row["total_ms"] / max(row["count"], 1)

        edges = [
            {"from": src, "to": dst, "count": count}
            for (src, dst), count in sorted(hierarchy_edges.items(), key=lambda item: item[1], reverse=True)
        ]

        return {
            "unique_modules": len(by_path),
            "total_module_calls": len(module_events),
            "top_modules_by_time": top,
            "hierarchy_edges": edges,
        }

    def _summarize_ops(self, aten_ops: List[AtenOpEvent]) -> Dict[str, Any]:
        by_name: Dict[str, Dict[str, Any]] = {}
        for ev in aten_ops:
            slot = by_name.setdefault(ev.op_name, {"op_name": ev.op_name, "count": 0, "total_ms": 0.0})
            slot["count"] += 1
            slot["total_ms"] += ev.duration_ms

        top = sorted(by_name.values(), key=lambda row: row["count"], reverse=True)
        for row in top:
            row["avg_ms"] = row["total_ms"] / max(row["count"], 1)

        return {
            "total_ops": len(aten_ops),
            "unique_ops": len(by_name),
            "top_ops": top,
        }
