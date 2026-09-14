# Slurm Compose

[![Lint](https://github.com/activatedgeek/slurm-compose/actions/workflows/prek.yml/badge.svg)](https://github.com/activatedgeek/slurm-compose/actions/workflows/prek.yml) [![Test](https://github.com/activatedgeek/slurm-compose/actions/workflows/pytest.yml/badge.svg)](https://github.com/activatedgeek/slurm-compose/actions/workflows/pytest.yml)

## Starter specification

`slurm_compose.spec` defines the new, declarative configuration API. It is a
standalone validation model; the existing CLI still uses the legacy `jobs`/`steps`
format. See [the example specification](tests/configs/spec.yml).

- `ComposeConfig` describes one job allocation, an optional shared metadata
  `directory`, and a mapping of named `steps`. The top-level `name` is a plain
  string, while `directory` is a path. Both are currently optional and may be
  populated by later validators. If `name` is omitted, it defaults to
  `SCOMPOSE_PROJECT_NAME` or the current working directory name.
- `sbatch` is a `SbatchConfig`: a flat block matching the sbatch argument fields of
  the existing `SlurmJob`, including `extra_argv`. Script-level orchestration
  settings are not sbatch options and do not belong in this block. The key is
  optional; when omitted, the execution layer can interpret the steps as standard
  Bash commands without explicit sbatch arguments.
- `sbatch.job_name: ${{ vars.NAME }}` explicitly references a top-level variable,
  as shown in the example. Use `${{ vars.NAME }}` for all substitutions. References
  are resolved recursively in every string, including embedded references such
  as `/models/${{ vars.MODEL }}`. A whole-value reference preserves scalar types,
  so `${{ vars.NODES }}` can populate an integer field. Other namespaces are
  rejected.
- Each entry in `steps` is a `StepConfig`. Its optional nested `kwargs` block is an
  `SrunConfig` or `PyxisConfig`, matching the arguments of `SrunScript` and
  `PyxisScript`. Container settings use `container_image`, `container_mounts`,
  `container_workdir`, and `container_mount_home` inside `kwargs`.
- A missing `kwargs` block is a shell step. Otherwise, the `kwargs` class supplies
  its `launcher` property, which is `srun` for both `SrunConfig` and its
  `PyxisConfig` subclass. Pyxis remains distinguishable by its concrete class
  and container fields. An empty `kwargs: {}` is invalid because it still requires
  explicit resources. User scripts own multinode coordination.
- The optional `type` field is reserved for custom `SrunConfig` subclasses. It
  accepts a fully qualified class name in the same format as
  `pkgutil.resolve_name`, for example `my_package.launchers:CustomSrunConfig`.
- Dependencies use `started`, `healthy`, or `completed_successfully` startup
  gates. A `healthy` dependency requires an enabled health check on its target.
- `HealthcheckConfig` uses Compose-style `test`, `interval`, `timeout`, `retries`,
  `start_period`, `start_interval`, and `disable` fields. Tests accept a shell
  string, `["CMD", executable, ...]`, `["CMD-SHELL", command]`, or `["NONE"]`.
  Durations use units such as `500ms`, `30s`, and `1m30s`. An enabled check must
  supply a test; image health checks are not implicitly inherited.
- The example health check tests a marker under `$SCOMPOSE_ARTIFACT_DIR`, the
  planned attempt-local artifact directory. Health-check execution and this
  runtime environment contract will be implemented with the supervisor.
- Top-level `completion` defaults to `mode: any`. Use `mode: all` to wait for
  all steps, or `mode: step` with a named `step` to finish when that step
  exits and stop supporting steps. Launcher
  class describes execution, not whether a command is finite or long-running.
- `script` is an ordered list of complete shell commands, not an argv list.
  Each item may be a multiline YAML block (`|`), including shell control flow or
  heredocs. Empty or whitespace-only items are rejected; contents are preserved.
  The command sequence belongs to one step, including for srun/Pyxis launches.
  Health-check `CMD` lists retain Compose's argv semantics. Environment values
  are strings.
- Resource fields are required in `sbatch` and every `kwargs` block. There is no
  automatic sizing or resource inheritance.

Load YAML with `ruamel.yaml`, then validate with Pydantic:

```python
from pathlib import Path

from ruamel.yaml import YAML
from slurm_compose.spec import ComposeConfig

yaml = YAML(typ="safe")
yaml.allow_duplicate_keys = False
config = ComposeConfig.model_validate(yaml.load(Path("compose.slurm.yaml").read_text()))

json_text = config.model_dump_json(indent=2)
yaml_data = config.model_dump(mode="json", exclude_none=True)
json_schema = ComposeConfig.model_json_schema()
```

Unknown fields, missing resources, invalid dependency references, dependency
cycles, and invalid completion targets are rejected. Substitution, retries, and
runtime supervision will be added separately.

## License

MIT
