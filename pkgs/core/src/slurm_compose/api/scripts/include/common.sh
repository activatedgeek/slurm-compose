#!/usr/bin/env bash


function safe_echo {
    mask_vars=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--string)
                mask_str="${2}"
                shift 2
            ;;
            -m|--mask-variables)
                IFS=',' read -ra mask_vars <<< "${2}"
                shift 2
            ;;
            *)
                shift 1
            ;;
        esac
    done

    ## Also mask any other protected env vars.
    for env_row in $(env | grep -E 'TOKEN|API_KEY|PASSWORD'); do
        # shellcheck disable=SC2207
        mask_vars+=($(echo "${env_row}" | cut -d= -f1))
    done

    for mask_var in "${mask_vars[@]}"; do
        mask_str="$(echo "$mask_str" | sed -e "s|\\\$${mask_var}|*******|g" -e "s|\\\$\\\{${mask_var}\\\}|*******|g" -e "s|${!mask_var}|*******|g")"
    done

    echo "${mask_str}"
    unset mask_vars mask_var mask_str
}


function install_uv {
    if ! command -v uv > /dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/bin" sh
    fi
}


function get_free_port {
    install_uv >&2

    uv run -q --no-progress --no-project python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}


function retry_until {
    ## max_retries (60) x delay (10s) = max_time (10 mins).
    local retries=12 max_retries=60 delay=10
    local cmd=()

    while [[ $# -gt 0 ]]; do
        case "${1}" in
            -r|--retries) retries="${2}"; shift 2 ;;
            -m|--max-retries) max_retries="${2}"; shift 2 ;;
            -d|--delay) delay="${2}"; shift 2 ;;
            --) shift; cmd=("${@}"); break ;;
            *) cmd+=("${1}"); shift ;;
        esac
    done

    if [[ ${#cmd[@]} -eq 0 ]]; then
        echo "[ERROR] retry_until requires a command." >&2
        return 1
    fi

    if [[ "${retries}" -eq -1 ]]; then
        retries="${max_retries}"
    fi

    until "${cmd[@]}"; do
        status=$?
        retries=$(( retries - 1 ))

        if [[ "${retries}" -lt 0 ]]; then
            echo "[ERROR] Command failed or timed out: ${cmd[*]}" >&2
            return "${status}"
        fi

        sleep "${delay}"
    done
}


function ray_health_check {
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        install_uv >&2
    fi

    local address client_address version py_version

    if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/HEAD" > /dev/null 2>&1; then
            echo "[ERROR] Ray head file ${RAY_RUNTIME_DIR}/HEAD not found." >&2
            return 1
        fi

        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/CLIENT" > /dev/null 2>&1; then
            echo "[ERROR] Ray client file ${RAY_RUNTIME_DIR}/CLIENT not found." >&2
            return 1
        fi

        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Ray version file ${RAY_RUNTIME_DIR}/VERSION not found." >&2
            return 1
        fi

        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/PY_VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Python version file ${RAY_RUNTIME_DIR}/PY_VERSION not found." >&2
            return 1
        fi

        address="$(cat "${RAY_RUNTIME_DIR}/HEAD")"
        client_address="$(cat "${RAY_RUNTIME_DIR}/CLIENT")"
        version="$(cat "${RAY_RUNTIME_DIR}/VERSION")"
        py_version="$(cat "${RAY_RUNTIME_DIR}/PY_VERSION")"
    fi

    while [[ $# -gt 0 ]]; do
        case "${1}" in
            --address) address="${2}"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ (-z "${address}") || (-z "${client_address}") || (-z "${version}") || (-z "${py_version}") ]]; then
        echo "[ERROR] Unable to construct ray address/version. Set RAY_RUNTIME_DIR." >&2
        return 1
    fi

    ray_cmd=(ray health-check --address "${address}")
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        ray_cmd=(uvx -q --no-progress -p "${py_version}" -w "ray[cgraph,default]==${version}" "${ray_cmd[@]}")
    fi

    if ! retry_until -r -1 -- "${ray_cmd[@]}" > /dev/null 2>&1; then
        echo "[ERROR] Ray health check failed or timed out." >&2
        return 1
    fi

    echo "[INFO] Ray head ${address} ready!" >&2

    if [[ -z "${RAY_SKIP_KV_INIT}" ]]; then
        if ! ray_kv init; then
            echo "[ERROR] Unable to initialize ray KV store." >&2
            return 1
        fi
    fi
}


function ray_kv {
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        install_uv >&2
    fi

    local address version py_version
    local actor_namespace="${RAY_ACTOR_NAMESPACE:-slurm-${SLURM_JOB_ID}}" actor_name="${RAY_ACTOR_NAME:-global_kv}"

    if [[ -n "${RAY_RUNTIME_DIR}" ]]; then
        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/CLIENT" > /dev/null 2>&1; then
            echo "[ERROR] Ray client file ${RAY_RUNTIME_DIR}/CLIENT not found." >&2
            return 1
        fi

        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Ray version file ${RAY_RUNTIME_DIR}/VERSION not found." >&2
            return 1
        fi

        if ! retry_until -r -1 -- test -f "${RAY_RUNTIME_DIR}/PY_VERSION" > /dev/null 2>&1; then
            echo "[ERROR] Python version file ${RAY_RUNTIME_DIR}/PY_VERSION not found." >&2
            return 1
        fi

        address="$(cat "${RAY_RUNTIME_DIR}/CLIENT")"
        version="$(cat "${RAY_RUNTIME_DIR}/VERSION")"
        py_version="$(cat "${RAY_RUNTIME_DIR}/PY_VERSION")"
    fi

    if [[ (-z "${address}") || (-z "${version}") || (-z "${py_version}") ]]; then
        echo "[ERROR] Unable to construct ray client address/version. Set RAY_RUNTIME_DIR." >&2
        return 1
    fi

    kv_cmd=(
        python "$(dirname "${BASH_SOURCE[0]}")/../ray/kv.py"
        --address "${address}" --namespace "${actor_namespace}" --name "${actor_name}"
        "${@}"
    )
    if [[ -z "${RAY_SKIP_UV_WRAP}" ]]; then
        kv_cmd=(uv run -q --no-progress --no-project -p "${py_version}" -w "ray[cgraph,default]==${version}" "${kv_cmd[@]}")
    fi

    "${kv_cmd[@]}"
}
