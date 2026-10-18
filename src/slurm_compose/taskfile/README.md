# taskfile

A Pydantic model subpackage of slurm-compose for a limited subset of Taskfile version 3.
This package does not install Task, load YAML, resolve templates, generate scripts,
or execute commands.

```python
from slurm_compose.taskfile import Taskfile

workflow = Taskfile.model_validate({
    "version": "3",
    "env": {"DATA_ROOT": "/shared/data"},
    "tasks": {
        "prepare": {"cmds": ["python prepare.py"]},
        "train": {
            "deps": ["prepare"],
            "env": {"MODEL": "qwen3"},
            "cmds": ["python train.py"],
        },
        "all": {"deps": ["train"]},
    },
})

data = workflow.model_dump(exclude_unset=True)
```

Supported fields are `version`, `tasks`, global and task-level scalar `vars`,
string-valued `env`, and task-level `desc`, `dir`, `cmds`, and `deps`.
Only full task mappings, shell command strings, and dependency names are supported.
Unknown fields, missing dependencies, and dependency cycles are rejected.
Commands are intended to be finite; the model cannot establish whether a shell
command will actually terminate. Task remains responsible for execution semantics,
including parallel dependencies and sequential commands.

Run the tests from the repository root with `python -m pytest tests/test_taskfile.py`.
