#!/usr/bin/env bash

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../include/common.sh"

if [[ -z "${RAY_RUNTIME_DIR}" ]]; then
    echo "[ERROR] Missing RAY_RUNTIME_DIR."
    exit 1
fi
unset RAY_ADDRESS TMPDIR RAY_TMPDIR

## https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html#system-configuration
if [[ $(ulimit -Hn) == "unlimited" ]] || [[ $(ulimit -Hn) -lt 65535 ]]; then
    if ! ulimit -Sn 65535; then
        echo "[WARNING] Could not set ulimit (old value $(ulimit -Hn))"
    fi
fi

ray_bin=(ray)
py_bin=(python3)
if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
    install_uv

    RAY_VERSION=${RAY_VERSION:-2.54.1}
    PY_VERSION=${PY_VERSION:-3.12.13}

    ray_bin=(uvx -q --no-progress -p "${PY_VERSION}" -w "ray[cgraph,default]==${RAY_VERSION}" "${ray_bin[@]}")
    py_bin=(uv run -q --no-progress --no-project -p "${PY_VERSION}" -w "ray[cgraph,default]==${RAY_VERSION}" "${py_bin[@]}")
fi

## All clients talking to this ray cluster must install using these exact versions.
RAY_VERSION=$("${ray_bin[@]}" --version | cut -d' ' -f3)
PY_VERSION=$("${py_bin[@]}" --version | cut -d' ' -f2)

if [[ $SLURM_PROCID -eq 0 ]]; then
    RAY_PORT=${RAY_PORT:-$(get_free_port)}
    RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-$(get_free_port)}

    export TMPDIR="${RAY_RUNTIME_DIR}/tmp-${SLURM_PROCID}"
    export RAY_TMPDIR="${TMPDIR}"
    echo "[INFO] Ray tmp dir set to ${RAY_TMPDIR}."

    echo "[INFO] Starting ray head: IPAddress=$(hostname --ip-address) Port=${RAY_PORT}"
    "${ray_bin[@]}" start \
        --block \
        --head \
        --include-dashboard=false \
        --disable-usage-stats \
        --object-manager-port 0 \
        --node-manager-port 0 \
        --metrics-export-port 0 \
        --dashboard-agent-grpc-port 0 \
        --min-worker-port 64000 \
        --max-worker-port 64999 \
        --node-ip-address "$(hostname --ip-address)" \
        --ray-client-server-port "${RAY_CLIENT_PORT}" \
        --port "${RAY_PORT}" "${@}" & ray_pid=$!

    RAY_ADDRESS="$(hostname --ip-address):${RAY_PORT}"
    RAY_CLIENT_ADDRESS="$(hostname --ip-address):${RAY_CLIENT_PORT}"

    mkdir -p "${RAY_RUNTIME_DIR}"

    echo "${RAY_ADDRESS}" > "${RAY_RUNTIME_DIR}/HEAD"
    echo "[INFO] Ray head Addr=${RAY_ADDRESS} written to ${RAY_RUNTIME_DIR}/HEAD."

    echo "${RAY_CLIENT_ADDRESS}" > "${RAY_RUNTIME_DIR}/CLIENT"
    echo "[INFO] Ray client Addr=${RAY_CLIENT_ADDRESS} written to ${RAY_RUNTIME_DIR}/CLIENT."

    echo "${RAY_VERSION}" > "${RAY_RUNTIME_DIR}/VERSION"
    echo "[INFO] Ray version ${RAY_VERSION} written to ${RAY_RUNTIME_DIR}/VERSION."

    echo "${PY_VERSION}" > "${RAY_RUNTIME_DIR}/PY_VERSION"
    echo "[INFO] Python version ${PY_VERSION} written to ${RAY_RUNTIME_DIR}/PY_VERSION."

    if ! ray_health_check; then
        exit 1
    fi

    wait $ray_pid
else
    if ! ray_health_check; then
        exit 1
    fi

    RAY_ADDRESS=$(cat "${RAY_RUNTIME_DIR}/HEAD")

    echo "[INFO] Connecting ray worker to head ${RAY_ADDRESS}..."
    "${ray_bin[@]}" start \
        --block \
        --address "${RAY_ADDRESS}" "${@}"
fi
