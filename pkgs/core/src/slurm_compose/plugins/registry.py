from importlib.metadata import entry_points

from .base import SlurmComposePlugin


class PluginRegistry:
    def __init__(self, group: str):
        self.group = group
        self._plugins = self._discover_plugins()

    def _discover_plugins(self) -> dict[str, SlurmComposePlugin]:
        plugins = {}
        for ep in entry_points(group=f"slurm_compose.{self.group}"):
            for plugin in ep.load()():
                if plugin.name in plugins:
                    raise ValueError(f"Duplicate plugin name '{plugin.name}' in group 'slurm_compose.{self.group}'")
                plugins[plugin.name] = plugin
        return plugins

    @property
    def all(self):
        return self._plugins.keys()

    def get(self, name: str) -> SlurmComposePlugin:
        if name not in self._plugins:
            raise ValueError(f"Missing plugin name '{name}' in group 'slurm_compose.{self.group}'")

        return self._plugins[name]


script_plugins = PluginRegistry("scripts")
export_plugins = PluginRegistry("export")
