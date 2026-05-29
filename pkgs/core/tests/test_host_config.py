from pathlib import Path

import pytest
from slurm_compose.api.exporter import SlurmSSHRemote


def test_empty_host():
    with pytest.raises(AssertionError):
        SlurmSSHRemote.load_config("empty", config_file=Path(__file__).parent / "test_hosts.toml")


def test_host():
    config = SlurmSSHRemote.load_config("test", config_file=Path(__file__).parent / "test_hosts.toml")

    assert config.get("hostname") == "example.com"
    assert config.get("user") == "test"
    assert config.get("identityfile") == "/etc/config/ssh.id"
    assert config.get("home_dir") == "/home/test/.slurm-compose"
    assert config.get("sbatch").get("bin") == "sbatch"
