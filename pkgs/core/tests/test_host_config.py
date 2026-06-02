from pathlib import Path

import pytest
from slurm_compose.api.exporter import SlurmSSHRemote


@pytest.fixture
def hosts_config_file() -> Path:
    return Path(__file__).parent / "test_hosts.toml"


def test_empty_host(hosts_config_file: Path):
    with pytest.raises(AssertionError):
        SlurmSSHRemote.load_config("empty", config_file=hosts_config_file)


def test_host(hosts_config_file: Path):
    config = SlurmSSHRemote.load_config("test", config_file=hosts_config_file)

    assert config.get("hostname") == "example.com"
    assert config.get("user") == "test"
    assert config.get("identityfile") == "/etc/config/ssh.id"
    assert config.get("home_dir") == "/home/test/.slurm-compose"
    assert config.get("sbatch").get("bin") == "sbatch"


def test_example(hosts_config_file: Path):
    config = SlurmSSHRemote.load_config("example", config_file=hosts_config_file)

    assert config.get("hostname") == "example"
    assert config.get("home_dir") == "/home/user/store/.slurm-compose"
    assert config.get("sbatch").get("bin") == "/home/user/.local/bin/sbatch"
    assert config.get("sbatch").get("account") == "example_account"
    assert config.get("sbatch").get("gpus_per_node") == 4
    for p in ["cpu", "gpu"]:
        assert isinstance(config.get("sbatch").get("partitions")[p], dict)
        assert isinstance(config.get("sbatch").get("partitions")[p + "_interactive"], dict)
