"""Resolution of the deliberately small Slurm Compose variable syntax."""

import re
from collections.abc import Mapping
from typing import Any

EXPRESSION = re.compile(r"\$\{\{\s*([^{}]+?)\s*\}\}")
VARIABLE = re.compile(r"vars\.([A-Za-z_][A-Za-z0-9_]*)\Z")


def resolve_vars(value: Any, variables: Mapping[str, Any]) -> Any:
    """Resolve ``${{ vars.NAME }}`` recursively without reparsing YAML.

    A whole-value reference preserves the variable's scalar type. Embedded
    references always produce strings. Dictionary keys are intentionally not
    resolved.
    """

    resolving: list[str] = []
    resolved: dict[str, Any] = {}

    def resolve_variable(name: str) -> Any:
        if name in resolving:
            cycle = " -> ".join([*resolving, name])
            raise ValueError(f"variable reference cycle: {cycle}")
        if name not in variables:
            raise ValueError(f"unknown variable {name!r}")
        if name not in resolved:
            resolving.append(name)
            resolved[name] = visit(variables[name])
            resolving.pop()
        return resolved[name]

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            matches = list(EXPRESSION.finditer(item))
            if not matches:
                return item

            if len(matches) == 1 and matches[0].span() == (0, len(item)):
                expression = matches[0].group(1).strip()
                match = VARIABLE.fullmatch(expression)
                if not match:
                    raise ValueError(f"unsupported variable expression {expression!r}")
                return resolve_variable(match.group(1))

            def replace(match: re.Match[str]) -> str:
                expression = match.group(1).strip()
                variable = VARIABLE.fullmatch(expression)
                if not variable:
                    raise ValueError(f"unsupported variable expression {expression!r}")
                resolved_value = resolve_variable(variable.group(1))
                if isinstance(resolved_value, (dict, list)):
                    raise ValueError(f"embedded variable {expression!r} must resolve to a scalar")
                return str(resolved_value)

            return EXPRESSION.sub(replace, item)

        if isinstance(item, Mapping):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, tuple):
            return tuple(visit(child) for child in item)
        return item

    return visit(value)
