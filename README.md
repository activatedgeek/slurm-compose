# Slurm Compose

[![Lint](https://github.com/activatedgeek/slurm-compose/actions/workflows/prek.yml/badge.svg)](https://github.com/activatedgeek/slurm-compose/actions/workflows/prek.yml) [![Test](https://github.com/activatedgeek/slurm-compose/actions/workflows/pytest.yml/badge.svg)](https://github.com/activatedgeek/slurm-compose/actions/workflows/pytest.yml)

## Taskfile workflow

`stask` validates a limited Taskfile specification and executes a generated copy
with Task. See [the debug Taskfile](tests/configs/Taskfile.yml).

```bash
uv run stask --taskfile tests/configs/Taskfile.yml
```

The legacy `slurm-compose` CLI continues to use the `jobs`/`steps` format.

## License

MIT
