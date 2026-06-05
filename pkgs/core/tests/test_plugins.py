import pytest
from slurm_compose.api.scripts import PyxisScript, Script, SrunScript
from slurm_compose.api.scripts.ray import IdleRayScript, RayScript
from slurm_compose.api.scripts.sglang import SGLangScript
from slurm_compose.api.scripts.vllm import vLLMScript
from slurm_compose.plugins import SlurmComposeExportPlugin, SlurmComposeScriptPlugin
from slurm_compose.plugins.export import DefaultExportPlugin
from slurm_compose.plugins.registry import export_plugins, script_plugins


@pytest.mark.parametrize(
    "name, cls",
    [
        ("command", Script),
        ("srun", SrunScript),
        ("pyxis", PyxisScript),
        ("ray", RayScript),
        ("idleray", IdleRayScript),
        ("sglang", SGLangScript),
        ("vllm", vLLMScript),
    ],
)
def test_core_script_plugin_registry(name, cls):
    plugin = script_plugins.get(name)

    assert issubclass(plugin, SlurmComposeScriptPlugin)
    assert plugin.name == name
    assert plugin.cls is cls


@pytest.mark.parametrize(
    "name, cls",
    [
        ("default", DefaultExportPlugin),
    ],
)
def test_core_export_plugin_registry(name, cls):
    plugin = export_plugins.get(name)

    assert issubclass(plugin, SlurmComposeExportPlugin)
    assert plugin.name == name
