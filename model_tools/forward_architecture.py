from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CONTAINER_MODULE_TYPES = {
    "ModuleList",
    "nn.ModuleList",
    "torch.nn.ModuleList",
    "Sequential",
    "nn.Sequential",
    "torch.nn.Sequential",
}

ACTIVATION_NAMES = {
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "softmax",
    "dropout",
}

LIBRARY_NAMESPACE_PREFIXES = (
    "torch",
    "F",
    "nn.functional",
    "torch.nn.functional",
    "math",
    "np",
    "numpy",
)


def safe_unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return f"<{node.__class__.__name__}>"


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return f"{dotted_name(node.value)}[{safe_unparse(node.slice)}]"
    return safe_unparse(node)


def sanitize_mermaid_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def parse_dim_list(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def shape_to_text(shape: Optional[List[str]]) -> str:
    if not shape:
        return "?"
    return "(" + ", ".join(shape) + ")"


def parse_bool_expr(expr: str, default: bool = True) -> bool:
    value = expr.strip()
    if not value:
        return default
    if value in {"False", "0", "None"}:
        return False
    if value in {"True", "1"}:
        return True
    return default


def parse_int_expr(expr: str) -> Optional[int]:
    try:
        return int(expr.strip())
    except Exception:
        return None


def is_library_namespace(name: str) -> bool:
    if name in LIBRARY_NAMESPACE_PREFIXES:
        return True
    return any(name.startswith(prefix + ".") for prefix in LIBRARY_NAMESPACE_PREFIXES)


def flatten_targets(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple):
        names: List[str] = []
        for elt in target.elts:
            names.extend(flatten_targets(elt))
        return names
    if isinstance(target, ast.List):
        names: List[str] = []
        for elt in target.elts:
            names.extend(flatten_targets(elt))
        return names
    if isinstance(target, ast.Attribute):
        return [safe_unparse(target)]
    if isinstance(target, ast.Subscript):
        return [safe_unparse(target)]
    return [safe_unparse(target)]


def self_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


class InputNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute):
            owner = node.func.value
            owner_name = dotted_name(owner)
            if owner_name != "self" and not owner_name.startswith("self.") and not is_library_namespace(owner_name):
                self.visit(owner)
        elif isinstance(node.func, ast.Subscript):
            self.visit(node.func.value)

        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.value is not None:
                self.visit(kw.value)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if isinstance(node.ctx, ast.Load):
            full = dotted_name(node)
            if full.startswith("self."):
                self.names.add(full)
                return
        self.generic_visit(node)


def collect_input_names(node: ast.AST) -> List[str]:
    collector = InputNameCollector()
    collector.visit(node)
    return sorted(collector.names)


def parse_forward_arguments(forward_fn: ast.FunctionDef) -> List[str]:
    args = [arg.arg for arg in forward_fn.args.args if arg.arg != "self"]
    if forward_fn.args.vararg:
        args.append(f"*{forward_fn.args.vararg.arg}")
    if forward_fn.args.kwarg:
        args.append(f"**{forward_fn.args.kwarg.arg}")
    return args


@dataclass
class SubmoduleSpec:
    attr_name: str
    constructor: str
    type_name: str
    args: List[str]
    kwargs: Dict[str, str]
    param_shapes: Dict[str, List[str]] = field(default_factory=dict)
    container_element_types: List[str] = field(default_factory=list)

    @property
    def is_container(self) -> bool:
        return self.constructor in CONTAINER_MODULE_TYPES or self.type_name in CONTAINER_MODULE_TYPES

    def primary_element_type(self) -> Optional[str]:
        if self.container_element_types:
            return self.container_element_types[0]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attr_name": self.attr_name,
            "constructor": self.constructor,
            "type_name": self.type_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "param_shapes": self.param_shapes,
            "container_element_types": self.container_element_types,
        }


@dataclass
class AliasBinding:
    source: str
    module_attr: Optional[str]
    module_type: str


@dataclass
class Operation:
    op_id: str
    kind: str
    name: str
    scope: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    expression: str = ""
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, str] = field(default_factory=dict)
    target: Optional[str] = None
    module_attr: Optional[str] = None
    module_type: Optional[str] = None
    inferred_output_shapes: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "name": self.name,
            "scope": self.scope,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "expression": self.expression,
            "args": self.args,
            "kwargs": self.kwargs,
            "target": self.target,
            "module_attr": self.module_attr,
            "module_type": self.module_type,
            "inferred_output_shapes": self.inferred_output_shapes,
        }


@dataclass
class ForwardAnalysis:
    class_name: str
    args: List[str]
    operations: List[Operation]
    data_flow_edges: List[Dict[str, str]]
    return_values: List[str]
    inferred_variable_shapes: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "args": self.args,
            "operations": [op.to_dict() for op in self.operations],
            "data_flow_edges": self.data_flow_edges,
            "return_values": self.return_values,
            "inferred_variable_shapes": self.inferred_variable_shapes,
        }


@dataclass
class ClassRecord:
    name: str
    bases: List[str]
    node: ast.ClassDef
    methods: Dict[str, ast.FunctionDef]
    submodules: Dict[str, SubmoduleSpec] = field(default_factory=dict)

    def has_forward(self) -> bool:
        return "forward" in self.methods


def extract_call_type_names(node: ast.AST) -> List[str]:
    names: List[str] = []

    if isinstance(node, ast.Call):
        names.append(dotted_name(node.func).split(".")[-1])
        return names

    if isinstance(node, ast.List):
        for elt in node.elts:
            names.extend(extract_call_type_names(elt))
        return names

    if isinstance(node, ast.Tuple):
        for elt in node.elts:
            names.extend(extract_call_type_names(elt))
        return names

    if isinstance(node, ast.ListComp):
        names.extend(extract_call_type_names(node.elt))
        return names

    if isinstance(node, ast.GeneratorExp):
        names.extend(extract_call_type_names(node.elt))
        return names

    return names


def infer_parameter_shapes(type_name: str, args: List[str], kwargs: Dict[str, str]) -> Dict[str, List[str]]:
    t = type_name.split(".")[-1].lower()

    def get_arg(index: int, key: str, default: str = "?") -> str:
        if index < len(args):
            return args[index]
        return kwargs.get(key, default)

    if "linear" == t:
        in_features = get_arg(0, "in_features")
        out_features = get_arg(1, "out_features")
        bias = get_arg(2, "bias", kwargs.get("bias", "True"))
        result = {"weight": [out_features, in_features]}
        if parse_bool_expr(bias, default=True):
            result["bias"] = [out_features]
        return result

    if "embedding" == t:
        num_embeddings = get_arg(0, "num_embeddings")
        embedding_dim = get_arg(1, "embedding_dim")
        return {"weight": [num_embeddings, embedding_dim]}

    if t.startswith("conv"):
        in_channels = get_arg(0, "in_channels")
        out_channels = get_arg(1, "out_channels")
        kernel_size = get_arg(2, "kernel_size", "?")
        dims = parse_dim_list(kernel_size)
        if not dims:
            dims = [kernel_size]
        groups = kwargs.get("groups", "1")
        weight = [out_channels, f"{in_channels}/{groups}"] + dims
        result = {"weight": weight}
        bias = kwargs.get("bias", "True")
        if parse_bool_expr(bias, default=True):
            result["bias"] = [out_channels]
        return result

    if t == "layernorm":
        normalized_shape = get_arg(0, "normalized_shape")
        dims = parse_dim_list(normalized_shape)
        if not dims:
            dims = [normalized_shape]
        return {"weight": dims, "bias": dims}

    if t.startswith("batchnorm"):
        num_features = get_arg(0, "num_features")
        return {"weight": [num_features], "bias": [num_features]}

    if t == "groupnorm":
        num_channels = get_arg(1, "num_channels")
        return {"weight": [num_channels], "bias": [num_channels]}

    return {}


def infer_module_output_shape(spec: SubmoduleSpec, input_shape: Optional[List[str]]) -> Optional[List[str]]:
    module_name = spec.type_name.split(".")[-1].lower()
    if input_shape is None:
        return None

    if module_name == "linear":
        out_features = spec.kwargs.get("out_features", spec.args[1] if len(spec.args) > 1 else "?")
        if not input_shape:
            return [out_features]
        return input_shape[:-1] + [out_features]

    if module_name == "embedding":
        embedding_dim = spec.kwargs.get("embedding_dim", spec.args[1] if len(spec.args) > 1 else "?")
        return input_shape + [embedding_dim]

    if module_name.startswith("conv1d"):
        out_channels = spec.kwargs.get("out_channels", spec.args[1] if len(spec.args) > 1 else "?")
        if len(input_shape) >= 3:
            return [input_shape[0], out_channels, "L_out"]
        return [out_channels, "L_out"]

    if module_name.startswith("conv2d"):
        out_channels = spec.kwargs.get("out_channels", spec.args[1] if len(spec.args) > 1 else "?")
        if len(input_shape) >= 4:
            return [input_shape[0], out_channels, "H_out", "W_out"]
        return [out_channels, "H_out", "W_out"]

    if module_name.startswith("conv3d"):
        out_channels = spec.kwargs.get("out_channels", spec.args[1] if len(spec.args) > 1 else "?")
        if len(input_shape) >= 5:
            return [input_shape[0], out_channels, "D_out", "H_out", "W_out"]
        return [out_channels, "D_out", "H_out", "W_out"]

    if module_name in {"layernorm", "batchnorm1d", "batchnorm2d", "batchnorm3d", "dropout"}:
        return list(input_shape)

    if any(name in module_name for name in ("relu", "gelu", "silu", "tanh", "sigmoid", "softmax")):
        return list(input_shape)

    return None


class ForwardExtractor:
    def __init__(self, class_record: ClassRecord) -> None:
        self.class_record = class_record
        self.operations: List[Operation] = []
        self.return_values: List[str] = []
        self._counter = 0

    def extract(self, forward_fn: ast.FunctionDef) -> ForwardAnalysis:
        args = parse_forward_arguments(forward_fn)
        self._walk_statements(forward_fn.body, scope="forward", aliases={})
        edges = self._build_data_flow_edges(args)
        return ForwardAnalysis(
            class_name=self.class_record.name,
            args=args,
            operations=self.operations,
            data_flow_edges=edges,
            return_values=self.return_values,
            inferred_variable_shapes={},
        )

    def _next_op_id(self) -> str:
        self._counter += 1
        return f"op_{self._counter:04d}"

    def _walk_statements(self, statements: List[ast.stmt], scope: str, aliases: Dict[str, AliasBinding]) -> None:
        local_aliases = dict(aliases)
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                outputs: List[str] = []
                for target in stmt.targets:
                    outputs.extend(flatten_targets(target))
                op = self._operation_from_value(stmt.value, outputs, scope, local_aliases)
                if op is not None:
                    self.operations.append(op)
                local_aliases = self._update_alias_map(local_aliases, outputs, stmt.value)
                continue

            if isinstance(stmt, ast.AnnAssign):
                outputs = flatten_targets(stmt.target)
                if stmt.value is not None:
                    op = self._operation_from_value(stmt.value, outputs, scope, local_aliases)
                    if op is not None:
                        self.operations.append(op)
                    local_aliases = self._update_alias_map(local_aliases, outputs, stmt.value)
                continue

            if isinstance(stmt, ast.AugAssign):
                outputs = flatten_targets(stmt.target)
                op = Operation(
                    op_id=self._next_op_id(),
                    kind="aug_assign",
                    name=stmt.op.__class__.__name__,
                    scope=scope,
                    inputs=collect_input_names(stmt.value) + collect_input_names(stmt.target),
                    outputs=outputs,
                    expression=safe_unparse(stmt),
                )
                self.operations.append(op)
                for name in outputs:
                    local_aliases.pop(name, None)
                continue

            if isinstance(stmt, ast.Expr):
                value = stmt.value
                if isinstance(value, ast.Call):
                    op = self._operation_from_call(value, [], scope, local_aliases)
                    self.operations.append(op)
                else:
                    op = Operation(
                        op_id=self._next_op_id(),
                        kind="expression",
                        name=value.__class__.__name__,
                        scope=scope,
                        inputs=collect_input_names(value),
                        expression=safe_unparse(value),
                    )
                    self.operations.append(op)
                continue

            if isinstance(stmt, ast.Return):
                return_outputs = self._extract_return_values(stmt.value)
                self.return_values.extend(return_outputs)
                op = Operation(
                    op_id=self._next_op_id(),
                    kind="return",
                    name="return",
                    scope=scope,
                    inputs=collect_input_names(stmt.value) if stmt.value is not None else [],
                    outputs=return_outputs,
                    expression=safe_unparse(stmt.value) if stmt.value is not None else "",
                )
                self.operations.append(op)
                continue

            if isinstance(stmt, ast.If):
                cond_op = Operation(
                    op_id=self._next_op_id(),
                    kind="control_if",
                    name="if_test",
                    scope=scope,
                    inputs=collect_input_names(stmt.test),
                    expression=safe_unparse(stmt.test),
                )
                self.operations.append(cond_op)
                self._walk_statements(stmt.body, f"{scope}/if_true[{safe_unparse(stmt.test)}]", dict(local_aliases))
                self._walk_statements(stmt.orelse, f"{scope}/if_false[{safe_unparse(stmt.test)}]", dict(local_aliases))
                continue

            if isinstance(stmt, ast.For):
                loop_op = Operation(
                    op_id=self._next_op_id(),
                    kind="control_for",
                    name="for_loop",
                    scope=scope,
                    inputs=collect_input_names(stmt.iter),
                    outputs=flatten_targets(stmt.target),
                    expression=f"for {safe_unparse(stmt.target)} in {safe_unparse(stmt.iter)}",
                )
                self.operations.append(loop_op)
                loop_aliases = self._infer_loop_aliases(stmt, local_aliases)
                body_aliases = dict(local_aliases)
                body_aliases.update(loop_aliases)
                self._walk_statements(stmt.body, f"{scope}/for_body[{safe_unparse(stmt.target)}]", body_aliases)
                self._walk_statements(stmt.orelse, f"{scope}/for_else[{safe_unparse(stmt.target)}]", dict(local_aliases))
                continue

            if isinstance(stmt, ast.While):
                while_op = Operation(
                    op_id=self._next_op_id(),
                    kind="control_while",
                    name="while_test",
                    scope=scope,
                    inputs=collect_input_names(stmt.test),
                    expression=safe_unparse(stmt.test),
                )
                self.operations.append(while_op)
                self._walk_statements(stmt.body, f"{scope}/while_body", dict(local_aliases))
                self._walk_statements(stmt.orelse, f"{scope}/while_else", dict(local_aliases))
                continue

            if isinstance(stmt, ast.With):
                with_op = Operation(
                    op_id=self._next_op_id(),
                    kind="control_with",
                    name="with_context",
                    scope=scope,
                    inputs=collect_input_names(stmt),
                    expression=safe_unparse(stmt),
                )
                self.operations.append(with_op)
                self._walk_statements(stmt.body, f"{scope}/with_body", dict(local_aliases))
                continue

            if isinstance(stmt, ast.Try):
                try_op = Operation(
                    op_id=self._next_op_id(),
                    kind="control_try",
                    name="try_block",
                    scope=scope,
                    expression="try",
                )
                self.operations.append(try_op)
                self._walk_statements(stmt.body, f"{scope}/try_body", dict(local_aliases))
                for handler in stmt.handlers:
                    self._walk_statements(handler.body, f"{scope}/except_body[{safe_unparse(handler.type)}]", dict(local_aliases))
                self._walk_statements(stmt.orelse, f"{scope}/try_else", dict(local_aliases))
                self._walk_statements(stmt.finalbody, f"{scope}/try_finally", dict(local_aliases))
                continue

    def _operation_from_value(
        self,
        value: ast.AST,
        outputs: List[str],
        scope: str,
        aliases: Dict[str, AliasBinding],
    ) -> Optional[Operation]:
        if isinstance(value, ast.Call):
            return self._operation_from_call(value, outputs, scope, aliases)
        return Operation(
            op_id=self._next_op_id(),
            kind="assign_expression",
            name=value.__class__.__name__,
            scope=scope,
            inputs=collect_input_names(value),
            outputs=outputs,
            expression=safe_unparse(value),
        )

    def _operation_from_call(
        self,
        call: ast.Call,
        outputs: List[str],
        scope: str,
        aliases: Dict[str, AliasBinding],
    ) -> Operation:
        kind = "function_call"
        name = dotted_name(call.func)
        target: Optional[str] = None
        module_attr: Optional[str] = None
        module_type: Optional[str] = None

        # self.submodule(...)
        if isinstance(call.func, ast.Attribute):
            owner = call.func.value
            if isinstance(owner, ast.Name) and owner.id == "self":
                attr = call.func.attr
                if attr in self.class_record.submodules:
                    spec = self.class_record.submodules[attr]
                    kind = "module_call"
                    name = f"self.{attr}"
                    target = f"self.{attr}"
                    module_attr = attr
                    module_type = spec.type_name
                else:
                    kind = "self_method_call"
                    name = f"self.{attr}"
                    target = "self"
            else:
                owner_name = dotted_name(owner)
                if is_library_namespace(owner_name):
                    kind = "function_call"
                    name = dotted_name(call.func)
                else:
                    kind = "tensor_method"
                    target = safe_unparse(owner)
                    name = call.func.attr

        # self.module_list[i](...)
        if isinstance(call.func, ast.Subscript) and isinstance(call.func.value, ast.Attribute):
            owner = call.func.value
            if isinstance(owner.value, ast.Name) and owner.value.id == "self":
                attr = owner.attr
                if attr in self.class_record.submodules:
                    spec = self.class_record.submodules[attr]
                    kind = "module_call"
                    target = f"self.{attr}[{safe_unparse(call.func.slice)}]"
                    name = target
                    module_attr = attr
                    module_type = spec.primary_element_type() or spec.type_name

        # alias-based module call: layer_module(...)
        if isinstance(call.func, ast.Name) and call.func.id in aliases:
            alias = aliases[call.func.id]
            kind = "module_call"
            name = call.func.id
            target = alias.source
            module_attr = alias.module_attr
            module_type = alias.module_type

        args = [safe_unparse(arg) for arg in call.args]
        kwargs = {kw.arg if kw.arg is not None else "**kwargs": safe_unparse(kw.value) for kw in call.keywords}
        inputs = collect_input_names(call)
        if kind == "tensor_method" and isinstance(call.func, ast.Attribute):
            receiver = safe_unparse(call.func.value)
            if receiver and (receiver.startswith("self.") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", receiver)):
                if receiver not in inputs:
                    inputs.append(receiver)
                inputs = sorted(set(inputs))

        return Operation(
            op_id=self._next_op_id(),
            kind=kind,
            name=name,
            scope=scope,
            inputs=inputs,
            outputs=outputs,
            expression=safe_unparse(call),
            args=args,
            kwargs=kwargs,
            target=target,
            module_attr=module_attr,
            module_type=module_type,
        )

    def _extract_return_values(self, value: Optional[ast.AST]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, ast.Name):
            return [value.id]
        if isinstance(value, ast.Tuple):
            outputs: List[str] = []
            for elt in value.elts:
                outputs.extend(self._extract_return_values(elt))
            return outputs
        if isinstance(value, ast.List):
            outputs: List[str] = []
            for elt in value.elts:
                outputs.extend(self._extract_return_values(elt))
            return outputs
        return [safe_unparse(value)]

    def _update_alias_map(
        self,
        aliases: Dict[str, AliasBinding],
        outputs: List[str],
        value: ast.AST,
    ) -> Dict[str, AliasBinding]:
        new_aliases = dict(aliases)
        inferred = self._infer_alias(value, new_aliases)
        simple_outputs = [name for name in outputs if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name)]

        if inferred is not None:
            for out_name in simple_outputs:
                new_aliases[out_name] = inferred
        else:
            for out_name in simple_outputs:
                new_aliases.pop(out_name, None)
        return new_aliases

    def _infer_alias(self, value: ast.AST, aliases: Dict[str, AliasBinding]) -> Optional[AliasBinding]:
        if isinstance(value, ast.Name):
            return aliases.get(value.id)

        if isinstance(value, ast.Attribute):
            attr = self_attr_name(value)
            if attr and attr in self.class_record.submodules:
                spec = self.class_record.submodules[attr]
                return AliasBinding(source=f"self.{attr}", module_attr=attr, module_type=spec.type_name)

        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute):
            attr = self_attr_name(value.value)
            if attr and attr in self.class_record.submodules:
                spec = self.class_record.submodules[attr]
                module_type = spec.primary_element_type() or spec.type_name
                return AliasBinding(
                    source=f"self.{attr}[{safe_unparse(value.slice)}]",
                    module_attr=attr,
                    module_type=module_type,
                )

        if isinstance(value, ast.Call):
            # getattr(self, "encoder")
            if isinstance(value.func, ast.Name) and value.func.id == "getattr":
                if len(value.args) >= 2 and isinstance(value.args[0], ast.Name) and value.args[0].id == "self":
                    if isinstance(value.args[1], ast.Constant) and isinstance(value.args[1].value, str):
                        attr = value.args[1].value
                        if attr in self.class_record.submodules:
                            spec = self.class_record.submodules[attr]
                            return AliasBinding(source=f"self.{attr}", module_attr=attr, module_type=spec.type_name)

        return None

    def _infer_loop_aliases(
        self,
        stmt: ast.For,
        aliases: Dict[str, AliasBinding],
    ) -> Dict[str, AliasBinding]:
        result: Dict[str, AliasBinding] = {}

        iter_expr = stmt.iter
        use_last_target_name = False

        if isinstance(iter_expr, ast.Call) and dotted_name(iter_expr.func).split(".")[-1] in {"enumerate", "reversed"}:
            if iter_expr.args:
                iter_expr = iter_expr.args[0]
            use_last_target_name = True

        inferred_binding: Optional[AliasBinding] = None

        if isinstance(iter_expr, ast.Attribute):
            attr = self_attr_name(iter_expr)
            if attr and attr in self.class_record.submodules:
                spec = self.class_record.submodules[attr]
                element_type = spec.primary_element_type()
                if element_type:
                    inferred_binding = AliasBinding(
                        source=f"self.{attr}[*]",
                        module_attr=attr,
                        module_type=element_type,
                    )

        if isinstance(iter_expr, ast.Name) and iter_expr.id in aliases:
            alias = aliases[iter_expr.id]
            if alias.module_attr and alias.module_attr in self.class_record.submodules:
                spec = self.class_record.submodules[alias.module_attr]
                element_type = spec.primary_element_type()
                if element_type:
                    inferred_binding = AliasBinding(
                        source=f"self.{alias.module_attr}[*]",
                        module_attr=alias.module_attr,
                        module_type=element_type,
                    )

        if inferred_binding is None:
            return result

        target_names = flatten_targets(stmt.target)
        simple_names = [name for name in target_names if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name)]
        if not simple_names:
            return result

        target_name = simple_names[-1] if use_last_target_name and len(simple_names) > 1 else simple_names[0]
        result[target_name] = inferred_binding
        return result

    def _build_data_flow_edges(self, args: List[str]) -> List[Dict[str, str]]:
        producers: Dict[str, str] = {arg: f"input::{arg}" for arg in args if not arg.startswith("*")}
        edges: List[Dict[str, str]] = []

        for op in self.operations:
            for inp in op.inputs:
                source = producers.get(inp)
                if source is not None:
                    edges.append({"from": source, "to": op.op_id, "tensor": inp})
            for out in op.outputs:
                if out:
                    producers[out] = op.op_id
        return edges


class ForwardArchitectureAnalyzer:
    def analyze_source(
        self,
        source_path: str | Path,
        entry_class: Optional[str] = None,
        input_shapes: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        source = Path(source_path)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        classes = self._collect_class_records(tree)

        for class_record in classes.values():
            class_record.submodules = self._extract_submodules(class_record)

        forward_analyses: Dict[str, ForwardAnalysis] = {}
        for class_name, class_record in classes.items():
            if "forward" not in class_record.methods:
                continue
            extractor = ForwardExtractor(class_record)
            analysis = extractor.extract(class_record.methods["forward"])
            self._propagate_shapes(analysis, class_record, input_shapes or {})
            forward_analyses[class_name] = analysis

        selected_entry = entry_class or self._select_entry_class(classes)
        if selected_entry not in classes:
            known = ", ".join(sorted(classes.keys()))
            raise ValueError(f"Entry class '{selected_entry}' was not found. Available classes: {known}")

        nested_graph = self._build_nested_forward_graph(classes, forward_analyses)

        report = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_path": str(source.resolve()),
            "entry_class": selected_entry,
            "input_shapes": input_shapes or {},
            "classes": {
                name: {
                    "bases": record.bases,
                    "has_forward": record.has_forward(),
                    "submodules": {k: v.to_dict() for k, v in record.submodules.items()},
                }
                for name, record in classes.items()
            },
            "forwards": {name: analysis.to_dict() for name, analysis in forward_analyses.items()},
            "nested_forward_graph": nested_graph,
        }
        return report

    def write_outputs(
        self,
        report: Dict[str, Any],
        output_dir: str | Path,
        max_ops_in_plot: int = 140,
    ) -> Dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        nested_mermaid = self.render_nested_mermaid(report)
        nested_mermaid_path = output_path / "nested_forward_architecture.mmd"
        nested_mermaid_path.write_text(nested_mermaid, encoding="utf-8")

        entry_class = report["entry_class"]
        dataflow_mermaid = self.render_dataflow_mermaid(report, entry_class=entry_class, max_ops=max_ops_in_plot)
        dataflow_mermaid_path = output_path / f"{entry_class}_dataflow.mmd"
        dataflow_mermaid_path.write_text(dataflow_mermaid, encoding="utf-8")

        markdown = self.render_markdown_bundle(report, nested_mermaid, dataflow_mermaid, entry_class)
        markdown_path = output_path / "architecture_report.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        return {
            "report_json": report_path,
            "nested_mermaid": nested_mermaid_path,
            "dataflow_mermaid": dataflow_mermaid_path,
            "markdown_report": markdown_path,
        }

    def render_nested_mermaid(self, report: Dict[str, Any]) -> str:
        entry_class = report["entry_class"]
        edges = report["nested_forward_graph"]

        reachable = {entry_class}
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge["from"] in reachable and edge["to"] not in reachable:
                    reachable.add(edge["to"])
                    changed = True

        lines = ["flowchart TD"]
        lines.append("  classDef module fill:#e8f0fe,stroke:#2563eb,color:#1e3a8a;")

        for class_name in sorted(reachable):
            node_id = sanitize_mermaid_id(class_name)
            lines.append(f'  {node_id}["{class_name}.forward"]:::module')

        for edge in edges:
            if edge["from"] not in reachable:
                continue
            src = sanitize_mermaid_id(edge["from"])
            dst = sanitize_mermaid_id(edge["to"])
            via = edge["via"].replace('"', "'")
            lines.append(f'  {src} -->|"{via}"| {dst}')

        if len(lines) == 2:
            # only header + classDef
            node_id = sanitize_mermaid_id(entry_class)
            lines.append(f'  {node_id}["{entry_class}.forward"]:::module')

        return "\n".join(lines) + "\n"

    def render_dataflow_mermaid(
        self,
        report: Dict[str, Any],
        entry_class: str,
        max_ops: int = 140,
    ) -> str:
        forward = report["forwards"].get(entry_class)
        if not forward:
            return "flowchart LR\n  missing[No forward() analysis available]\n"

        operations = forward["operations"]
        op_count = len(operations)
        truncated = False
        if op_count > max_ops:
            operations = operations[:max_ops]
            truncated = True

        op_ids = {op["op_id"] for op in operations}
        lines = ["flowchart LR"]
        lines.append("  classDef op fill:#eef2ff,stroke:#4f46e5,color:#312e81;")
        lines.append("  classDef io fill:#ecfdf5,stroke:#059669,color:#065f46;")

        input_shapes = report.get("input_shapes", {})
        for arg in forward["args"]:
            if arg.startswith("*"):
                continue
            node_id = f"in_{sanitize_mermaid_id(arg)}"
            shape = shape_to_text(input_shapes.get(arg))
            lines.append(f'  {node_id}(["{arg}\\n{shape}"]):::io')

        for op in operations:
            node_id = sanitize_mermaid_id(op["op_id"])
            label = self._format_operation_label(op)
            lines.append(f'  {node_id}["{label}"]:::op')

        for edge in forward["data_flow_edges"]:
            if edge["to"] not in op_ids:
                continue
            dst = sanitize_mermaid_id(edge["to"])
            if edge["from"].startswith("input::"):
                arg = edge["from"].split("::", 1)[1]
                src = f"in_{sanitize_mermaid_id(arg)}"
            else:
                if edge["from"] not in op_ids:
                    continue
                src = sanitize_mermaid_id(edge["from"])
            tensor = edge["tensor"].replace('"', "'")
            lines.append(f'  {src} -->|"{tensor}"| {dst}')

        for op in operations:
            if op["kind"] != "return":
                continue
            op_node = sanitize_mermaid_id(op["op_id"])
            outputs = op.get("outputs") or ["return"]
            for index, output_name in enumerate(outputs):
                out_id = f"out_{sanitize_mermaid_id(op['op_id'])}_{index}"
                shape = shape_to_text(op.get("inferred_output_shapes", {}).get(output_name))
                label = output_name.replace('"', "'")
                lines.append(f'  {out_id}(["return {label}\\n{shape}"]):::io')
                lines.append(f"  {op_node} --> {out_id}")

        if truncated:
            lines.append(
                f'  trunc_note["Diagram truncated to first {max_ops} operations out of {op_count}. '
                'See report.json for full analysis."]:::io'
            )

        return "\n".join(lines) + "\n"

    def render_markdown_bundle(
        self,
        report: Dict[str, Any],
        nested_mermaid: str,
        dataflow_mermaid: str,
        entry_class: str,
    ) -> str:
        lines = [
            "# Model Architecture Analysis",
            "",
            f"- Source: `{report['source_path']}`",
            f"- Entry class: `{entry_class}`",
            f"- Generated: `{report['generated_at']}`",
            "",
            "## Nested forward() architecture",
            "",
            "```mermaid",
            nested_mermaid.rstrip(),
            "```",
            "",
            f"## `{entry_class}.forward()` data-flow",
            "",
            "```mermaid",
            dataflow_mermaid.rstrip(),
            "```",
            "",
            "## Notes",
            "",
            "- Parameter dimensions are inferred from common layer constructors when possible.",
            "- Tensor dimensions are symbolic and based on static shape propagation from your provided `--input-shape` values.",
            "- For exact runtime shapes on dynamic paths, combine this report with a runtime trace.",
            "",
            "Machine-readable details are stored in `report.json`.",
            "",
        ]
        return "\n".join(lines)

    def _format_operation_label(self, op: Dict[str, Any]) -> str:
        bits = [op["op_id"], f'{op["kind"]}: {op["name"]}']
        outputs = op.get("outputs") or []
        shape_map = op.get("inferred_output_shapes", {})
        if outputs:
            shape_bits = []
            for out_name in outputs[:2]:
                shape_bits.append(f"{out_name}: {shape_to_text(shape_map.get(out_name))}")
            if shape_bits:
                bits.append(", ".join(shape_bits))
        return "\\n".join(part.replace('"', "'") for part in bits if part)

    def _collect_class_records(self, tree: ast.Module) -> Dict[str, ClassRecord]:
        classes: Dict[str, ClassRecord] = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods: Dict[str, ast.FunctionDef] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    methods[stmt.name] = stmt
            bases = [safe_unparse(base) for base in node.bases]
            classes[node.name] = ClassRecord(
                name=node.name,
                bases=bases,
                node=node,
                methods=methods,
            )
        return classes

    def _extract_submodules(self, class_record: ClassRecord) -> Dict[str, SubmoduleSpec]:
        init_fn = class_record.methods.get("__init__")
        if init_fn is None:
            return {}

        submodules: Dict[str, SubmoduleSpec] = {}

        def walk_statements(statements: Iterable[ast.stmt]) -> None:
            for stmt in statements:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        attr = self_attr_name(target)
                        if attr and isinstance(stmt.value, ast.Call):
                            spec = self._make_submodule_spec(attr, stmt.value)
                            if spec:
                                submodules[attr] = spec
                elif isinstance(stmt, ast.AnnAssign):
                    attr = self_attr_name(stmt.target)
                    if attr and isinstance(stmt.value, ast.Call):
                        spec = self._make_submodule_spec(attr, stmt.value)
                        if spec:
                            submodules[attr] = spec
                elif isinstance(stmt, ast.If):
                    walk_statements(stmt.body)
                    walk_statements(stmt.orelse)
                elif isinstance(stmt, ast.For):
                    walk_statements(stmt.body)
                    walk_statements(stmt.orelse)
                elif isinstance(stmt, ast.While):
                    walk_statements(stmt.body)
                    walk_statements(stmt.orelse)
                elif isinstance(stmt, ast.With):
                    walk_statements(stmt.body)
                elif isinstance(stmt, ast.Try):
                    walk_statements(stmt.body)
                    for handler in stmt.handlers:
                        walk_statements(handler.body)
                    walk_statements(stmt.orelse)
                    walk_statements(stmt.finalbody)

        walk_statements(init_fn.body)
        return submodules

    def _make_submodule_spec(self, attr: str, call: ast.Call) -> Optional[SubmoduleSpec]:
        constructor = dotted_name(call.func)
        if not constructor:
            return None
        type_name = constructor.split(".")[-1]
        args = [safe_unparse(arg) for arg in call.args]
        kwargs = {
            kw.arg if kw.arg is not None else "**kwargs": safe_unparse(kw.value)
            for kw in call.keywords
        }
        param_shapes = infer_parameter_shapes(type_name, args, kwargs)

        container_element_types: List[str] = []
        if constructor in CONTAINER_MODULE_TYPES or type_name in CONTAINER_MODULE_TYPES:
            for arg in call.args:
                container_element_types.extend(extract_call_type_names(arg))
            deduped: List[str] = []
            for name in container_element_types:
                if name not in deduped:
                    deduped.append(name)
            container_element_types = deduped

        return SubmoduleSpec(
            attr_name=attr,
            constructor=constructor,
            type_name=type_name,
            args=args,
            kwargs=kwargs,
            param_shapes=param_shapes,
            container_element_types=container_element_types,
        )

    def _select_entry_class(self, classes: Dict[str, ClassRecord]) -> str:
        with_forward = [record for record in classes.values() if record.has_forward()]
        if not with_forward:
            raise ValueError("No classes with forward() were found in the source file.")

        def has_base(record: ClassRecord, key: str) -> bool:
            return any(key.lower() in base.lower() for base in record.bases)

        for record in with_forward:
            if has_base(record, "PreTrainedModel"):
                return record.name
        for record in with_forward:
            if record.name.lower().endswith("model"):
                return record.name
        return with_forward[0].name

    def _build_nested_forward_graph(
        self,
        classes: Dict[str, ClassRecord],
        forward_analyses: Dict[str, ForwardAnalysis],
    ) -> List[Dict[str, str]]:
        edge_set: set[Tuple[str, str, str]] = set()

        # Prefer edges observed in forward() operations.
        for class_name, analysis in forward_analyses.items():
            for op in analysis.operations:
                if op.kind != "module_call" or not op.module_type:
                    continue
                target_type = op.module_type.split(".")[-1]
                if target_type not in classes or not classes[target_type].has_forward():
                    continue
                via = op.module_attr or op.name
                edge_set.add((class_name, target_type, via))

        # Fallback to __init__ wiring for modules that are defined but not directly called.
        for class_record in classes.values():
            if not class_record.has_forward():
                continue
            for attr_name, spec in class_record.submodules.items():
                candidate_types = [spec.type_name] + spec.container_element_types
                for candidate in candidate_types:
                    if candidate in classes and classes[candidate].has_forward():
                        edge_set.add((class_record.name, candidate, attr_name))

        edges = [{"from": src, "to": dst, "via": via} for src, dst, via in sorted(edge_set)]
        return edges

    def _propagate_shapes(
        self,
        analysis: ForwardAnalysis,
        class_record: ClassRecord,
        input_shapes: Dict[str, List[str]],
    ) -> None:
        env: Dict[str, List[str]] = {}
        for arg_name in analysis.args:
            if arg_name in input_shapes:
                env[arg_name] = list(input_shapes[arg_name])

        for op in analysis.operations:
            inferred = self._infer_operation_output_shapes(op, env, class_record)
            op.inferred_output_shapes = inferred
            for out_name, shape in inferred.items():
                env[out_name] = shape

        analysis.inferred_variable_shapes = env

    def _infer_operation_output_shapes(
        self,
        op: Operation,
        env: Dict[str, List[str]],
        class_record: ClassRecord,
    ) -> Dict[str, List[str]]:
        if not op.outputs:
            return {}

        if op.kind == "module_call" and op.module_attr in class_record.submodules:
            spec = class_record.submodules[op.module_attr]
            input_shape = self._first_known_input_shape(op.inputs, env)
            out_shape = infer_module_output_shape(spec, input_shape)
            if out_shape is not None:
                return {name: list(out_shape) for name in op.outputs}

        if op.kind == "tensor_method":
            receiver_shape = env.get(op.target or "")
            if receiver_shape is not None:
                inferred = self._infer_tensor_method_shape(op, receiver_shape)
                if inferred is not None:
                    return {name: list(inferred) for name in op.outputs}

        if op.kind == "function_call":
            inferred = self._infer_function_call_shape(op, env)
            if inferred is not None:
                return {name: list(inferred) for name in op.outputs}

        if op.kind == "assign_expression":
            copied = self._copy_shape_from_inputs(op.inputs, env)
            if copied is not None:
                return {name: list(copied) for name in op.outputs}

        if op.kind == "aug_assign":
            copied = self._copy_shape_from_inputs(op.inputs, env)
            if copied is not None:
                return {name: list(copied) for name in op.outputs}

        if op.kind == "return":
            inferred: Dict[str, List[str]] = {}
            for name in op.outputs:
                if name in env:
                    inferred[name] = list(env[name])
            return inferred

        return {}

    def _first_known_input_shape(
        self,
        inputs: List[str],
        env: Dict[str, List[str]],
    ) -> Optional[List[str]]:
        for name in inputs:
            if name in env:
                return env[name]
        return None

    def _copy_shape_from_inputs(
        self,
        inputs: List[str],
        env: Dict[str, List[str]],
    ) -> Optional[List[str]]:
        for name in inputs:
            if name in env:
                return env[name]
        return None

    def _infer_tensor_method_shape(self, op: Operation, base_shape: List[str]) -> Optional[List[str]]:
        method = op.name
        args = op.args
        kwargs = op.kwargs

        if method in {"to", "type", "float", "half", "double", "long", "int", "cpu", "cuda", "detach", "clone"}:
            return list(base_shape)

        if method in {"contiguous"}:
            return list(base_shape)

        if method in {"view", "reshape"}:
            dims = [dim for dim in args if dim]
            if dims:
                return dims
            return None

        if method in {"permute"}:
            order = [parse_int_expr(item) for item in args]
            if any(idx is None for idx in order):
                return None
            permuted: List[str] = []
            for idx in order:
                if idx is None:
                    return None
                resolved = idx if idx >= 0 else len(base_shape) + idx
                if resolved < 0 or resolved >= len(base_shape):
                    return None
                permuted.append(base_shape[resolved])
            return permuted

        if method in {"transpose", "swapaxes"}:
            if len(args) < 2:
                return None
            dim0 = parse_int_expr(args[0])
            dim1 = parse_int_expr(args[1])
            if dim0 is None or dim1 is None:
                return None
            shape = list(base_shape)
            dim0 = dim0 if dim0 >= 0 else len(shape) + dim0
            dim1 = dim1 if dim1 >= 0 else len(shape) + dim1
            if dim0 < 0 or dim1 < 0 or dim0 >= len(shape) or dim1 >= len(shape):
                return None
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            return shape

        if method == "unsqueeze":
            if not args:
                return None
            dim = parse_int_expr(args[0])
            if dim is None:
                return None
            shape = list(base_shape)
            insert_at = dim if dim >= 0 else len(shape) + dim + 1
            insert_at = max(0, min(insert_at, len(shape)))
            shape.insert(insert_at, "1")
            return shape

        if method == "squeeze":
            shape = list(base_shape)
            if args:
                dim = parse_int_expr(args[0])
                if dim is None:
                    return None
                resolved = dim if dim >= 0 else len(shape) + dim
                if 0 <= resolved < len(shape) and shape[resolved] == "1":
                    shape.pop(resolved)
                return shape
            return [d for d in shape if d != "1"]

        if method == "flatten":
            start_dim = parse_int_expr(kwargs.get("start_dim", args[0] if len(args) > 0 else "0"))
            end_dim = parse_int_expr(kwargs.get("end_dim", args[1] if len(args) > 1 else "-1"))
            if start_dim is None or end_dim is None:
                return None
            shape = list(base_shape)
            if start_dim < 0:
                start_dim += len(shape)
            if end_dim < 0:
                end_dim += len(shape)
            if not (0 <= start_dim < len(shape) and 0 <= end_dim < len(shape) and start_dim <= end_dim):
                return None
            merged = "*".join(shape[start_dim : end_dim + 1]) or "flat"
            return shape[:start_dim] + [merged] + shape[end_dim + 1 :]

        return None

    def _infer_function_call_shape(
        self,
        op: Operation,
        env: Dict[str, List[str]],
    ) -> Optional[List[str]]:
        call_name = op.name.split(".")[-1].lower()

        if call_name in ACTIVATION_NAMES:
            return self._copy_shape_from_inputs(op.inputs, env)

        if call_name in {"matmul", "bmm"}:
            if len(op.inputs) < 2:
                return None
            lhs = env.get(op.inputs[0])
            rhs = env.get(op.inputs[1])
            if lhs is None or rhs is None:
                return None
            if len(lhs) == 2 and len(rhs) == 2:
                return [lhs[0], rhs[1]]
            if len(lhs) >= 2 and len(rhs) >= 2:
                return lhs[:-1] + [rhs[-1]]
            return None

        if call_name == "cat":
            # Conservative: keep the first tensor shape if unknown concat dim/value.
            return self._copy_shape_from_inputs(op.inputs, env)

        if call_name == "stack":
            base = self._copy_shape_from_inputs(op.inputs, env)
            if base is None:
                return None
            dim_expr = op.args[1] if len(op.args) > 1 else op.kwargs.get("dim", "0")
            dim = parse_int_expr(dim_expr)
            if dim is None:
                return None
            shape = list(base)
            insert_at = dim if dim >= 0 else len(shape) + dim + 1
            insert_at = max(0, min(insert_at, len(shape)))
            shape.insert(insert_at, "N_stack")
            return shape

        return None


def parse_input_shape_overrides(raw_pairs: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for pair in raw_pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid --input-shape value '{pair}'. Expected format: name:dim1,dim2,...")
        name, dims = pair.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --input-shape value '{pair}'. Missing argument name before ':'.")
        parsed_dims = [dim.strip() for dim in dims.split(",") if dim.strip()]
        if not parsed_dims:
            raise ValueError(f"Invalid --input-shape value '{pair}'. Missing dimensions after ':'.")
        result[name] = parsed_dims
    return result
