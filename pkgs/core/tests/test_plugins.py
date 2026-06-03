import dataclasses

import pytest
from slurm_compose.api.scripts import PyxisScript, Script, SrunScript
from slurm_compose.api.scripts.ray import IdleRayScript, RayScript
from slurm_compose.api.scripts.sglang import SGLangScript
from slurm_compose.api.scripts.vllm import vLLMScript
from slurm_compose.plugins import SlurmComposeScriptPlugin
from slurm_compose.plugins.registry import script_plugins


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

    assert isinstance(plugin, SlurmComposeScriptPlugin)
    assert plugin.name == name
    assert plugin.cls is cls
    with pytest.raises(dataclasses.FrozenInstanceError):
        plugin.name = "fail_change"
