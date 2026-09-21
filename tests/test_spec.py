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
    PyxisConfig,
    SbatchConfig,
    SbatchJobConfig,
    SrunConfig,
    StepConfig,
    StepKwargsConfig,
)


class CustomSrunConfig(SrunConfig):
    custom_flag: str

    def script(self, command):
        return f"custom-srun {command}"


class CustomStepConfig(StepKwargsConfig):
    custom_flag: str

    def script(self, command):
        return f"custom-step {command}"


@pytest.fixture
def spec_data():
    yaml = YAML(typ="safe")
    yaml.allow_duplicate_keys = False
    return yaml.load((Path(__file__).parent / "configs/spec.yml").read_text())


def test_spec_round_trip(spec_data):
    config = ComposeConfig.model_validate(spec_data)
    assert config.schema_version == 1
    assert config.directory == STATE_HOME
    assert [step["name"] for step in SbatchJobConfig(compose=config).step_topology] == [
        "prepare",
        "model",
        "evaluate",
    ]
    job = SbatchJobConfig(compose=config)
    assert job.job_name == f"evaluation-{config.hash[:16]}"
    assert job.job_dir == STATE_HOME / job.job_name
    assert "topology" not in config.json()
    assert isinstance(config.sbatch.completion, CompletionConfig)
    assert isinstance(config.steps["model"].kwargs, PyxisConfig)
    assert isinstance(config.steps["evaluate"].kwargs, SrunConfig)
    assert config.steps["evaluate"].kwargs.gpus_per_node == 0
    assert config.steps["prepare"].kwargs is None
    assert ComposeConfig.model_validate_json(config.model_dump_json()) == config

    yaml = YAML(typ="safe")
    output = StringIO()
    yaml.dump(config.model_dump(mode="json", exclude_none=True), output)
    assert ComposeConfig.model_validate(yaml.load(output.getvalue())) == config
    assert config.hash == ComposeConfig.model_validate(spec_data).hash
    spec_data["name"] = "different"
    assert ComposeConfig.model_validate(spec_data).hash != config.hash
    schema = ComposeConfig.model_json_schema()
    assert "SbatchConfig" in schema["$defs"]
    assert "StepConfig" in schema["$defs"]
    assert "kwargs" in schema["$defs"]["StepConfig"]["properties"]
    assert "srun" not in schema["$defs"]["StepConfig"]["properties"]
    assert "sbatch" in schema["properties"]
    assert "job" not in schema["properties"]
    assert "steps" in schema["properties"]
    assert "services" not in schema["properties"]
    assert "directory" in schema["properties"]
    assert "runtime" not in schema["properties"]


def test_materialize_respects_topology_and_step_launchers(spec_data):
    spec_data["steps"]["prepare"]["before_script"] = ["setup"]
    spec_data["steps"]["prepare"]["after_script"] = ["cleanup"]
    config = ComposeConfig.model_validate(spec_data)
    script = SbatchJobConfig(
        compose=config,
        extra_argv=["--comment", "evaluation"],
    ).script()
    assert f"#SBATCH --job-name={SbatchJobConfig(compose=config).job_name}" in script
    assert f"SCOMPOSE_JOB_DIR={SbatchJobConfig(compose=config).job_dir}" in script
    assert f"#SBATCH --output={SbatchJobConfig(compose=config).output}" in script
    assert '--output="${SCOMPOSE_JOB_DIR}/slurm-%j/model.log"' in script
    assert "SCOMPOSE_JOB_NAME=evaluation" in script
    assert f"SCOMPOSE_JOB_HASH={config.hash}" in script
    assert 'SCOMPOSE_JOB_HASH_SHORT="${SCOMPOSE_JOB_HASH:0:16}"' in script
    assert script.index("# Step 0: prepare") < script.index("# Step 1: model") < script.index("# Step 2: evaluate")
    assert "setup\nbash -l 'python /shared/project/prepare.py'\ncleanup" in script
    assert "srun --nodes=1" in script
    assert "--container-image=/shared/images/model.sqsh" in script
    assert "declare -A step_pids=()" in script
    assert "} & step_pids[prepare]=$!" in script
    assert "} & step_pids[prepare]=$!\n\n\n{\n##\n# Step 1: model" in script
    assert 'wait "${step_pids[prepare]}"' in script
    assert 'wait "${step_pids[evaluate]}"' in script
    assert 'kill -TERM "$pid"' in script


def test_step_scriptv_and_srun_command_rendering(spec_data):
    spec_data["steps"]["prepare"].update(before_script=["setup"], after_script=["cleanup"])
    config = ComposeConfig.model_validate(spec_data)
    prepare = config.steps["prepare"]
    assert prepare.scriptv == ["setup", "bash -l 'python /shared/project/prepare.py'", "cleanup"]

    model = config.steps["model"]
    assert model.scriptv[0].startswith("srun --nodes=1")


@pytest.mark.parametrize("field", ["nodes", "ntasks_per_node", "cpus_per_task", "gpus_per_node"])
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
    "command",
    ["", " \n\t", "python\x00evaluate.py", [], [""], ["python", 3], ["python", "\x00"]],
)
def test_invalid_command(spec_data, command):
    spec_data["steps"]["evaluate"]["command"] = command
    with pytest.raises(ValidationError, match="command"):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("name", ["prepare", "model", "evaluate"])
def test_command_forms_and_multiline_strings_preserved(spec_data, name):
    command = ['python "a path with spaces.py" --job "$SLURM_JOB_ID"', "echo ready"]
    spec_data["steps"][name]["command"] = command
    config = ComposeConfig.model_validate(spec_data)
    assert config.steps[name].command == command
    assert ComposeConfig.model_validate_json(config.model_dump_json()).steps[name].command == command

    yaml = YAML(typ="safe")
    output = StringIO()
    yaml.dump(config.model_dump(mode="json", exclude_none=True), output)
    assert ComposeConfig.model_validate(yaml.load(output.getvalue())).steps[name].command == command


def test_command_shell_form_preserved(spec_data):
    command = "cd /shared/project\npython evaluate.py\n"
    spec_data["steps"]["evaluate"]["command"] = command
    assert ComposeConfig.model_validate(spec_data).steps["evaluate"].command == command


def test_step_scripts_are_lists_of_commands(spec_data):
    spec_data["steps"]["prepare"].update(
        before_script=["module load python", "mkdir -p /shared/run"],
        after_script=["rm -rf /shared/run"],
    )
    step = ComposeConfig.model_validate(spec_data).steps["prepare"]
    assert step.before_script == ["module load python", "mkdir -p /shared/run"]
    assert step.after_script == ["rm -rf /shared/run"]


@pytest.mark.parametrize("field", ["commmand", "resources", "kind", "readiness", "image", "step"])
def test_unknown_step_fields_rejected(spec_data, field):
    spec_data["steps"]["model"][field] = "invalid"
    with pytest.raises(ValidationError, match=field):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("field", ["resources", "scheduler", "env", "steps", "max_restarts"])
def test_sbatch_contains_only_submission_options(spec_data, field):
    spec_data["sbatch"][field] = {}
    with pytest.raises(ValidationError, match=field):
        ComposeConfig.model_validate(spec_data)


def test_unknown_dependency(spec_data):
    spec_data["steps"]["evaluate"]["depends_on"] = {"missing": {"condition": "completed_successfully"}}
    with pytest.raises(ValidationError, match="unknown step 'missing'"):
        ComposeConfig.model_validate(spec_data)


@pytest.mark.parametrize("target", ["evaluate", "model"])
def test_dependency_cycle(spec_data, target):
    spec_data["steps"]["model"]["depends_on"] = {target: {"condition": "completed_successfully"}}
    with pytest.raises(ValidationError, match="cycle"):
        ComposeConfig.model_validate(spec_data)


def test_invalid_completion_target(spec_data):
    spec_data["sbatch"]["completion"]["step"] = "missing"
    with pytest.raises(ValidationError, match="completion"):
        ComposeConfig.model_validate(spec_data)


def test_default_completion(spec_data):
    del spec_data["sbatch"]["completion"]
    assert ComposeConfig.model_validate(spec_data).sbatch.completion.mode == "any"


def test_schema_version_defaults_to_one(spec_data):
    assert ComposeConfig.model_validate(spec_data).schema_version == 1
    spec_data["schema_version"] = 1
    assert ComposeConfig.model_validate(spec_data).schema_version == 1


def test_unsupported_schema_version_rejected(spec_data):
    spec_data["schema_version"] = 2
    with pytest.raises(ValidationError, match="schema_version"):
        ComposeConfig.model_validate(spec_data)


def test_directory_is_optional(spec_data):
    spec_data.pop("directory", None)
    config = ComposeConfig.model_validate(spec_data)
    assert config.name == "evaluation"
    assert config.directory == STATE_HOME


def test_sbatch_is_optional(spec_data):
    spec_data.pop("sbatch")
    config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch is None


def test_name_uses_project_name_fallback(spec_data, monkeypatch):
    spec_data.pop("name", None)
    monkeypatch.setattr("slurm_compose.spec.config.PROJECT_NAME", "project-name")
    assert ComposeConfig.model_validate(spec_data).name == "project-name"


def test_name_is_required_without_project_name(spec_data, monkeypatch):
    spec_data.pop("name", None)
    monkeypatch.setattr("slurm_compose.spec.config.PROJECT_NAME", None)
    with pytest.raises(ValidationError, match="name is required"):
        ComposeConfig.model_validate(spec_data)


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


@pytest.mark.parametrize(
    "config_class,backend_class",
    [(SbatchConfig, SlurmJob), (SrunConfig, SrunScript), (PyxisConfig, PyxisScript)],
)
def test_argument_names_match_existing_backends(config_class, backend_class):
    argument_names = {field.name for field in fields(backend_class) if field.metadata.get("argv", True)}
    excluded = {"completion"}
    if config_class is SbatchConfig:
        excluded.add("max_restarts")
    if config_class is SbatchConfig:
        argument_names -= {"account", "partition", "qos", "begin"}
    expected = argument_names | {"extra_argv"}
    assert set(config_class.model_fields) - excluded == expected


def test_argument_configs_pass_directly_to_backends(spec_data):
    spec_data["sbatch"].update(array="0-2")
    spec_data["steps"]["model"]["kwargs"].update(
        container_mounts=["/shared:/shared"], container_workdir="/shared/project", container_mount_home=True
    )
    config = ComposeConfig.model_validate(spec_data)
    job = SlurmJob(**config.sbatch.model_dump(exclude_none=True, exclude={"completion"}))
    script = job.materialize()
    assert "#SBATCH --job-name=evaluation" in script
    assert "#SBATCH --array=0-2" in script

    for name, step in config.steps.items():
        if step.kwargs is None:
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


def test_sbatch_argv_uses_equals_and_appends_extra_args(spec_data):
    spec_data["sbatch"].update(requeue=True, extra_argv=["--comment=evaluation"])
    config = ComposeConfig.model_validate(spec_data).sbatch
    assert config.argv == [
        "--nodes=1",
        "--ntasks-per-node=2",
        "--cpus-per-task=4",
        "--gpus-per-node=1",
        "--mem=32G",
        "--job-name=evaluation",
        "--time=01:00:00",
        "--open-mode=append",
        "--requeue",
        "--comment=evaluation",
    ]


def test_srun_argv_is_available_from_resource_config():
    config = SrunConfig(nodes=1, ntasks_per_node=1, cpus_per_task=1, gpus_per_node=0, mem="1G")
    assert config.argv == [
        "--nodes=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=1",
        "--gpus-per-node=0",
        "--mem=1G",
        "--wait=10",
        "--kill-on-bad-exit=1",
        "--exact",
    ]


def test_pyxis_argv_joins_container_mounts():
    config = PyxisConfig(
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=1,
        gpus_per_node=0,
        container_image="image.sqsh",
        container_mounts=["/data:/data", "/models:/models"],
    )
    assert config.container_mounts == ["/data:/data", "/models:/models"]
    assert "--container-mounts=/data:/data,/models:/models" in config.argv


def test_pyxis_container_mounts_default_is_mutable():
    config = PyxisConfig(
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=1,
        gpus_per_node=0,
        container_image="image.sqsh",
    )
    config.container_mounts.append("/data:/data")
    assert "--container-mounts=/data:/data" in config.argv


def test_pyxis_argv_omits_empty_container_mounts():
    config = PyxisConfig(
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=1,
        gpus_per_node=0,
        container_image="image.sqsh",
    )
    assert not any(argument.startswith("--container-mounts") for argument in config.argv)


def test_invalid_sbatch_time_is_preserved_with_warning(spec_data):
    spec_data["sbatch"]["time"] = "not-a-duration"
    with pytest.warns(RuntimeWarning, match="could not parse sbatch time"):
        config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch.time == "not-a-duration"


@pytest.mark.parametrize("options", [{}])
def test_shell_step_without_srun(options):
    command = ["python", "path with spaces.py"]
    step = StepConfig(command=command, **options)
    assert step.kwargs is None
    assert step.kwargs is None
    assert step.command == command
    assert StepConfig.model_validate_json(step.model_dump_json()) == step


def test_empty_srun_is_not_a_shell_step():
    with pytest.raises(ValidationError, match="nodes"):
        StepConfig(kwargs={}, command=["true"])


def test_legacy_job_key_rejected(spec_data):
    spec_data["job"] = spec_data.pop("sbatch")
    with pytest.raises(ValidationError, match="job"):
        ComposeConfig.model_validate(spec_data)


def test_sbatch_job_name_defaults_to_compose_name(spec_data):
    config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch.job_name == "evaluation"
    assert config.name == "evaluation"
    assert config.sbatch.model_dump()["job_name"] == "evaluation"
    assert "job_name" in SbatchConfig.model_json_schema()["properties"]
    restored = ComposeConfig.model_validate_json(config.model_dump_json())
    assert restored.sbatch.job_name == "evaluation"

    assert config.sbatch.job_name == "evaluation"


def test_literal_sbatch_job_name(spec_data):
    spec_data["sbatch"]["job_name"] = "override"
    assert ComposeConfig.model_validate(spec_data).sbatch.job_name == "override"


def test_default_sbatch_job_name_reference(spec_data):
    assert ComposeConfig.model_validate(spec_data).sbatch.job_name == "evaluation"


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
        command=["./run.sh"],
    )
    assert isinstance(step.kwargs, CustomSrunConfig)


def test_custom_pydantic_type_is_resolved():
    step = StepConfig(
        type=f"{__name__}:CustomStepConfig",
        kwargs={"custom_flag": "enabled"},
        command=["./run.sh"],
    )
    assert isinstance(step.kwargs, CustomStepConfig)
    assert step.kwargs.custom_flag == "enabled"


def test_builtin_launcher_type_is_not_supported():
    with pytest.raises(ValidationError, match="unable to resolve custom type"):
        StepConfig(type="pyxis", kwargs=None, command=["./run.sh"])


def test_vars_resolve_full_and_embedded_values(spec_data):
    spec_data["vars"].update(IMAGE="model.sqsh", PATH="/shared/models")
    spec_data["steps"]["model"]["command"] = ["run", "--model", "${{ vars.PATH }}/${{ vars.MODEL }}"]
    config = ComposeConfig.model_validate(spec_data)
    assert config.sbatch.nodes == 1
    assert config.steps["model"].command == ["run", "--model", "/shared/models/qwen3"]


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
