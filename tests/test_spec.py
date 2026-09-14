from dataclasses import fields
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from slurm_compose.api.scripts import PyxisScript, SrunScript
from slurm_compose.api.slurm import SlurmJob
from slurm_compose.config import STATE_HOME
from slurm_compose.spec import (
    CompletionConfig,
    ComposeConfig,
    HealthcheckConfig,
    PyxisConfig,
    SbatchConfig,
    SrunConfig,
    StepConfig,
)


class CustomSrunConfig(SrunConfig):
    custom_flag: str


@pytest.fixture
def spec_data():
    yaml = YAML(typ="safe")
    yaml.allow_duplicate_keys = False
    return yaml.load((Path(__file__).parent / "configs/spec.yml").read_text())


def test_spec_round_trip(spec_data):
    config = ComposeConfig.model_validate(spec_data)
    assert config.schema_version == 1
    assert config.directory == Path("/shared/slurm-compose/evaluation")
    assert isinstance(config.completion, CompletionConfig)
    assert isinstance(config.steps["model"].kwargs, PyxisConfig)
    assert isinstance(config.steps["evaluate"].kwargs, SrunConfig)
    assert config.steps["model"].kwargs.launcher == "srun"
    assert config.steps["evaluate"].kwargs.launcher == "srun"
    assert config.steps["evaluate"].kwargs.gpus_per_node == 0
    assert config.steps["prepare"].launcher == "shell"
    assert config.steps["prepare"].kwargs is None
    assert ComposeConfig.model_validate_json(config.model_dump_json()) == config

    yaml = YAML(typ="safe")
    output = StringIO()
    yaml.dump(config.model_dump(mode="json", exclude_none=True), output)
    assert ComposeConfig.model_validate(yaml.load(output.getvalue())) == config
    schema = ComposeConfig.model_json_schema()
    assert "SbatchConfig" in schema["$defs"]
    assert "StepConfig" in schema["$defs"]
    assert "kwargs" in schema["$defs"]["StepConfig"]["properties"]
    assert "srun" not in schema["$defs"]["StepConfig"]["properties"]
    assert "sbatch" in schema["properties"]
    assert "job" not in schema["properties"]
    assert "HealthcheckConfig" in schema["$defs"]
    assert "steps" in schema["properties"]
    assert "services" not in schema["properties"]
    assert "directory" in schema["properties"]
    assert "runtime" not in schema["properties"]


@pytest.mark.parametrize("field", ["nodes", "ntasks_per_node", "cpus_per_task", "gpus_per_node", "mem"])
@pytest.mark.parametrize("scope", ["sbatch", "srun"])
def test_resources_must_be_explicit(spec_data, field, scope):
    target = spec_data["sbatch"] if scope == "sbatch" else spec_data["steps"]["evaluate"]["kwargs"]
    del target[field]
    with pytest.raises(ValidationError, match=field):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("nodes", [0, -1, "2", True])
def test_invalid_resource_counts(spec_data, nodes):
    spec_data["sbatch"]["nodes"] = nodes
    with pytest.raises(ValidationError, match="nodes"):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize(
    "script",
    [[], "python evaluate.py", [""], ["true", ""], [" \n\t"], ["python", 3], ["python", "\x00"]],
)
def test_invalid_script(spec_data, script):
    spec_data["steps"]["evaluate"]["script"] = script
    with pytest.raises(ValidationError, match="script"):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("name", ["prepare", "model", "evaluate"])
def test_complete_commands_and_multiline_scripts_preserved(spec_data, name):
    script = [
        'python "a path with spaces.py" --job "$SLURM_JOB_ID"',
        'if test -f ready; then\n  echo "Ready"\nfi\n',
        "python - <<'PY'\nprint('hello')\nPY\n",
    ]
    spec_data["steps"][name]["script"] = script
    config = ComposeConfig.model_validate(spec_data)
    assert config.steps[name].script == script
    assert ComposeConfig.model_validate_json(config.model_dump_json()).steps[name].script == script

    yaml = YAML(typ="safe")
    output = StringIO()
    yaml.dump(config.model_dump(mode="json", exclude_none=True), output)
    assert ComposeConfig.model_validate(yaml.load(output.getvalue())).steps[name].script == script


@pytest.mark.parametrize("field", ["commmand", "resources", "kind", "readiness", "image", "step"])
def test_unknown_step_fields_rejected(spec_data, field):
    spec_data["steps"]["model"][field] = "invalid"
    with pytest.raises(ValidationError, match=field):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("field", ["resources", "scheduler", "completion", "env", "steps", "max_restarts"])
def test_sbatch_contains_only_submission_options(spec_data, field):
    spec_data["sbatch"][field] = {}
    with pytest.raises(ValidationError, match=field):
        ComposeConfig.model_validate(spec_data)


def test_unknown_dependency(spec_data):
    spec_data["steps"]["evaluate"]["depends_on"] = {"missing": {"condition": "started"}}
    with pytest.raises(ValidationError, match="unknown step 'missing'"):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("target", ["evaluate", "model"])
def test_dependency_cycle(spec_data, target):
    spec_data["steps"]["model"]["depends_on"] = {target: {"condition": "started"}}
    with pytest.raises(ValidationError, match="cycle"):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("condition", ["started", "completed_successfully"])
def test_non_health_dependencies_need_no_healthcheck(spec_data, condition):
    spec_data["steps"]["evaluate"]["depends_on"]["model"]["condition"] = condition
    del spec_data["steps"]["model"]["healthcheck"]
    ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("healthcheck", [None, {"disable": True}, {"test": ["NONE"]}])
def test_healthy_dependency_requires_enabled_healthcheck(spec_data, healthcheck):
    spec_data["steps"]["model"]["healthcheck"] = healthcheck
    with pytest.raises(ValidationError, match="enabled healthcheck"):
        ComposeConfig.model_validate(spec_data)


def test_invalid_completion_target(spec_data):
    spec_data["completion"]["step"] = "missing"
    with pytest.raises(ValidationError, match="completion"):
        ComposeConfig.model_validate(spec_data)


def test_default_completion(spec_data):
    del spec_data["completion"]
    assert ComposeConfig.model_validate(spec_data).completion.mode == "any"


def test_schema_version_defaults_to_one(spec_data):
    assert ComposeConfig.model_validate(spec_data).schema_version == 1
    spec_data["schema_version"] = 1
    assert ComposeConfig.model_validate(spec_data).schema_version == 1


def test_unsupported_schema_version_rejected(spec_data):
    spec_data["schema_version"] = 2
    with pytest.raises(ValidationError, match="schema_version"):
        ComposeConfig.model_validate(spec_data)


def test_name_and_directory_are_optional(spec_data):
    spec_data.pop("name")
    spec_data.pop("directory")
    config = ComposeConfig.model_validate(spec_data)
    assert config.name == Path.cwd().name
    assert config.directory == STATE_HOME


def test_sbatch_is_optional(spec_data):
    spec_data.pop("sbatch")
    config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch is None


def test_name_defaults_to_project_name(spec_data, monkeypatch):
    spec_data.pop("name")
    monkeypatch.setattr("slurm_compose.spec.models.PROJECT_NAME", "my-project")
    assert ComposeConfig.model_validate(spec_data).name == "my-project"


def test_name_has_fallback_when_project_name_is_unset(spec_data, monkeypatch):
    spec_data.pop("name")
    monkeypatch.setattr("slurm_compose.spec.models.PROJECT_NAME", None)
    assert ComposeConfig.model_validate(spec_data).name == Path.cwd().name


@pytest.mark.parametrize("image", [None])
def test_invalid_container_image(spec_data, image):
    spec_data["steps"]["model"]["kwargs"]["container_image"] = image
    with pytest.raises(ValidationError, match="container_image"):
        ComposeConfig.model_validate(spec_data)


def test_pyxis_paths_are_normalized(spec_data):
    spec_data["steps"]["model"]["kwargs"].update(
        container_image="/shared/images/model.sqsh", container_workdir="/shared/project"
    )
    config = ComposeConfig.model_validate(spec_data)
    assert config.steps["model"].kwargs.container_image == Path("/shared/images/model.sqsh")
    assert config.steps["model"].kwargs.container_workdir == Path("/shared/project")


def test_launcher_is_fixed_and_not_serialized():
    config = SrunConfig(nodes=1, ntasks_per_node=1, cpus_per_task=1, gpus_per_node=0, mem="1G")
    assert config.launcher == "srun"
    assert "launcher" not in config.model_dump()
    with pytest.raises(ValidationError):
        config.launcher = "custom"

    pyxis = PyxisConfig(
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=1,
        gpus_per_node=0,
        mem="1G",
        container_image="image.sqsh",
    )
    assert pyxis.launcher == "srun"
    assert "launcher" not in pyxis.model_dump()


@pytest.mark.parametrize("test", ["test -f ready", ["CMD", "test", "-f", "ready"], ["CMD-SHELL", "test -f ready"]])
def test_healthcheck_commands(test):
    healthcheck = HealthcheckConfig(test=test, interval="1m30s", timeout="500ms")
    assert healthcheck.enabled
    assert healthcheck.test == test
    assert HealthcheckConfig.model_validate_json(healthcheck.model_dump_json()) == healthcheck


@pytest.mark.parametrize("test", ["", [], ["CMD"], ["NONE", "extra"], ["CMD-SHELL", "true", "extra"], ["true"]])
def test_invalid_healthcheck_commands(test):
    with pytest.raises(ValidationError, match="test"):
        HealthcheckConfig(test=test)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"test": "true", "retries": 0}],
)
def test_invalid_healthcheck_settings(kwargs):
    with pytest.raises(ValidationError):
        HealthcheckConfig(**kwargs)


def test_healthcheck_duration_strings_are_unvalidated():
    healthcheck = HealthcheckConfig(test="true", interval="tomorrow", timeout="custom-duration")
    assert healthcheck.interval == "tomorrow"
    assert healthcheck.timeout == "custom-duration"


@pytest.mark.parametrize(
    "config_class,backend_class",
    [(SbatchConfig, SlurmJob), (SrunConfig, SrunScript), (PyxisConfig, PyxisScript)],
)
def test_argument_names_match_existing_backends(config_class, backend_class):
    argument_names = {field.name for field in fields(backend_class) if field.metadata.get("argv", True)}
    assert set(config_class.model_fields) - {"launcher"} == argument_names | {"extra_argv"}


def test_argument_configs_pass_directly_to_backends(spec_data):
    spec_data["sbatch"].update(account="research", begin="now", array="0-2", extra_argv=["--comment=evaluation"])
    spec_data["steps"]["model"]["kwargs"].update(
        container_mounts=["/shared:/shared"], container_workdir="/shared/project", container_mount_home=True
    )
    config = ComposeConfig.model_validate(spec_data)
    job = SlurmJob(**config.sbatch.model_dump(exclude_none=True))
    script = job.materialize()
    assert "#SBATCH --account=research" in script
    assert "#SBATCH --job-name=evaluation" in script
    assert "#SBATCH --begin=now" in script
    assert "#SBATCH --array=0-2" in script
    assert "#SBATCH --comment=evaluation" in script

    for name, step in config.steps.items():
        if step.launcher == "shell":
            continue
        kwargs = step.kwargs.model_dump(exclude_none=True)
        kwargs.setdefault("job_name", name)
        backend = PyxisScript if isinstance(step.kwargs, PyxisConfig) else SrunScript
        # Backend commands are still argv; the new script list needs a file
        # when execution is integrated. Only the argument blocks map directly.
        launch = backend(command=["bash", "/generated/step.sh"], **kwargs)
        assert launch.argv[0] == "srun"
        assert "--nodes" in launch.argv
        if isinstance(step.kwargs, PyxisConfig):
            assert "--container-image" in launch.argv
            assert "/shared/images/model.sqsh" in launch.argv


def test_sbatch_time_is_normalized(spec_data):
    spec_data["sbatch"]["time"] = "1d 2h 3m 4s"
    assert ComposeConfig.model_validate(spec_data).sbatch.time == "1-02:03:04"


def test_invalid_sbatch_time_is_preserved_with_warning(spec_data):
    spec_data["sbatch"]["time"] = "not-a-duration"
    with pytest.warns(RuntimeWarning, match="could not parse sbatch time"):
        config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch.time == "not-a-duration"


@pytest.mark.parametrize("options", [{}])
def test_shell_step_without_srun(options):
    script = ['python "path with spaces.py"', 'echo "$SLURM_JOB_ID"']
    step = StepConfig(script=script, **options)
    assert step.launcher == "shell"
    assert step.kwargs is None
    assert step.script == script
    assert StepConfig.model_validate_json(step.model_dump_json()) == step


def test_empty_srun_is_not_a_shell_step():
    with pytest.raises(ValidationError, match="nodes"):
        StepConfig(kwargs={}, script=["true"])


def test_legacy_job_key_rejected(spec_data):
    spec_data["job"] = spec_data.pop("sbatch")
    with pytest.raises(ValidationError, match="job"):
        ComposeConfig.model_validate(spec_data)


def test_sbatch_job_name_resolves_from_vars(spec_data):
    assert spec_data["sbatch"]["job_name"] == "${{ vars.NAME }}"
    config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch.job_name == config.name == "evaluation"
    assert config.sbatch.model_dump()["job_name"] == "evaluation"
    assert "job_name" in SbatchConfig.model_json_schema()["properties"]
    assert spec_data["sbatch"]["job_name"] == "${{ vars.NAME }}"
    restored = ComposeConfig.model_validate_json(config.model_dump_json())
    assert restored.sbatch.job_name == config.name

    assert config.sbatch.job_name == "evaluation"


def test_literal_sbatch_job_name(spec_data):
    spec_data["sbatch"]["job_name"] = "override"
    assert ComposeConfig.model_validate(spec_data).sbatch.job_name == "override"


def test_default_sbatch_job_name_reference(spec_data):
    del spec_data["sbatch"]["job_name"]
    assert ComposeConfig.model_validate(spec_data).sbatch.job_name is None


def test_unknown_sbatch_job_name_reference(spec_data):
    spec_data["sbatch"]["job_name"] = "${{ name }}"
    with pytest.raises(ValidationError, match="unsupported variable expression"):
        ComposeConfig.model_validate(spec_data)


def test_custom_srun_type_is_resolved():
    step = StepConfig(
        type=f"{__name__}:CustomSrunConfig",
        kwargs={
            "nodes": 1,
            "ntasks_per_node": 1,
            "cpus_per_task": 1,
            "gpus_per_node": 0,
            "mem": "1G",
            "custom_flag": "enabled",
        },
        script=["./run.sh"],
    )
    assert isinstance(step.kwargs, CustomSrunConfig)
    assert step.kwargs.custom_flag == "enabled"
    assert step.launcher == "srun"


def test_builtin_launcher_type_is_not_supported():
    with pytest.raises(ValidationError, match="unable to resolve custom srun type"):
        StepConfig(type="pyxis", kwargs=None, script=["./run.sh"])


def test_vars_resolve_full_and_embedded_values(spec_data):
    spec_data["vars"].update(IMAGE="model.sqsh", PATH="/shared/models")
    spec_data["steps"]["model"]["script"] = ["run --model ${{ vars.PATH }}/${{ vars.MODEL }}"]
    config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch.nodes == 1
    assert config.steps["model"].script == ["run --model /shared/models/qwen3"]


@pytest.mark.parametrize("value", ["${{ vars.MISSING }}", "${{ vars.MISSING }}-suffix"])
def test_unknown_vars_rejected(spec_data, value):
    spec_data["sbatch"]["job_name"] = value
    with pytest.raises(ValidationError, match="unknown variable 'MISSING'"):
        ComposeConfig.model_validate(spec_data)


def test_var_reference_cycles_rejected(spec_data):
    spec_data["vars"].update(A="${{ vars.B }}", B="${{ vars.A }}")
    spec_data["sbatch"]["job_name"] = "${{ vars.A }}"
    with pytest.raises(ValidationError, match="variable reference cycle"):
        ComposeConfig.model_validate(spec_data)


def test_only_vars_namespace_is_supported(spec_data):
    spec_data["sbatch"]["job_name"] = "${{ name }}"
    with pytest.raises(ValidationError, match="unsupported variable expression"):
        ComposeConfig.model_validate(spec_data)
