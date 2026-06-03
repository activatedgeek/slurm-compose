import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Iterator, Type

from slurm_compose.api.scripts import Script

from .base import SlurmComposeExportPlugin, SlurmComposeScriptPlugin


def get_all_subclasses(cls: Type) -> Iterator[Type]:
    for subcls in cls.__subclasses__():
        yield subcls
        yield from get_all_subclasses(subcls)


def find_subclasses_in_package(package: ModuleType, cls: Type):
    ## Import only top-level __init__ modules (ensure *only* scripts are exposed).
    for info in pkgutil.iter_modules(package.__path__, prefix=f"{package.__name__}."):
        importlib.import_module(info.name)

    yield cls

    yield from get_all_subclasses(cls)


def find_subclasses_in_module(module: ModuleType, cls: Type):
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, cls) and cls is not obj:
            yield obj


def build_script_plugins(*cls: Script) -> Iterator[SlurmComposeScriptPlugin]:
    for c in cls:
        name = c.__name__.removesuffix("Script").lower() or "command"
        yield type(f"{name}Plugin", (SlurmComposeScriptPlugin,), {"name": name, "cls": c})


def register_core_script_plugins() -> Iterator[SlurmComposeScriptPlugin]:
    import slurm_compose.api.scripts as scripts_package

    yield from build_script_plugins(*find_subclasses_in_package(scripts_package, Script))


def register_core_export_plugins() -> Iterator[SlurmComposeExportPlugin]:
    import slurm_compose.plugins.export as export_package

    yield from find_subclasses_in_module(export_package, SlurmComposeExportPlugin)
