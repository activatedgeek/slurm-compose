from pathlib import Path
from typing import Any

from slurm_compose.config import logger

from .base import Script


def resolve_log_template(dir_or_file: str | Path, template: str) -> Path:
    if not dir_or_file:
        return

    dir_or_file = Path(dir_or_file)

    ## Assumes no file extension to be a folder.
    if not dir_or_file.suffix:
        dir_or_file = dir_or_file / template

    return dir_or_file.absolute()


def fields_to_argv(obj: Script, ignore_keys: str | None = None, equals_separated: bool = False):
    def _handle_arg(k):
        arg_name = k.replace("_", "-")
        arg_val = getattr(obj, k)

        if isinstance(arg_val, bool):
            if not arg_val:
                arg_name = "no-" + arg_name
            arg_val = None
        elif isinstance(arg_val, int):
            ...
        elif isinstance(arg_val, str):
            ...
        elif isinstance(arg_val, list):
            arg_val = ",".join([str(v) for v in arg_val])
        elif isinstance(arg_val, Path):
            arg_val = str(arg_val.absolute())
        else:
            raise ValueError(
                f"Unsupported value type {type(arg_val).__name__} for {k} ({arg_name}). Use only bool/int/str/Path."
            )

        arg_name = "--" + arg_name

        arg = list(filter(lambda a: a is not None, [arg_name, arg_val]))
        if equals_separated:
            arg = ["=".join(str(a) for a in arg)]

        return arg

    return sum(
        [
            _handle_arg(k)
            for k in type(obj).fields().keys()
            if k not in (ignore_keys or set())
            and getattr(obj, k) is not None
            and type(obj).__dataclass_fields__[k].metadata.get("argv", True)
        ],
        [],
    )


def maybe_update_fields(obj: Any, force: bool = False, **kwargs):
    """Updates attributes that are non-null, unless force is True.

    gpus_per_node are handled separately to allow for cases when
    it is deliberately set to 0.
    """

    for k, v in kwargs.items():
        if not hasattr(obj, k):
            continue

        old_v = getattr(obj, k)
        should_update = old_v is None and v is not None and v != old_v
        if force:
            should_update = True

        ## Override force behavior for this attribute.
        if k == "gpus_per_node":
            if v == -1:
                should_update = old_v != 0
            elif old_v == 0:
                should_update = False

        if should_update:
            logger.debug(f"Overriding job param {k} with value {v} (previously {old_v})")
            setattr(obj, k, v)
