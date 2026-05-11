from dataclasses import dataclass
from typing import ClassVar, Self, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf


@dataclass
class BaseArgs:
    version: ClassVar[int] = 1
    """Preserve version for backward/forward compatibility of `from_yaml` method when writing in `to_yaml`"""

    def to_dict(self) -> dict:
        ## NOTE: OmegaConf keeps None values, remove them.
        def _remove_none(d: dict):
            return {k: _remove_none(v) if isinstance(v, dict) else v for k, v in d.items() if v is not None}

        def _dataclass_to_dict(d: Self):
            _dict = _remove_none(OmegaConf.to_container(OmegaConf.structured(d), resolve=True, enum_to_str=True))
            ## NOTE: Remove empty dictionaries.
            return {k: v for k, v in _dict.items() if v}

        return _dataclass_to_dict(self)

    @classmethod
    def fields(cls) -> dict[str]:
        def _get_type(v):
            if get_origin(v) is ClassVar:
                return get_args(v)[0]
            return v

        return {k: _get_type(v) for k, v in get_type_hints(cls).items()}
