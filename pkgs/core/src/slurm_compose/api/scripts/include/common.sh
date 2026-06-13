#!/usr/bin/env bash


function safe_echo {
    local mask_vars=() mask_str

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
}


function install_uv {
    if ! command -v uv > /dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/bin" sh
    fi
}


function get_free_port {
    if ! install_uv > /dev/null 2>&1; then
        return 1
    fi

    uv run -q --no-progress --no-project python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}


function retry_until {
    ## max_retries (60) x delay (10s) = max_time (10 mins).
    local retries=12 max_retries=60 delay=10
    local cmd=() status

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
        return 1
    fi

    if [[ "${retries}" -eq -1 ]]; then
        retries="${max_retries}"
    fi

    until "${cmd[@]}"; do
        status=$?
        retries=$(( retries - 1 ))

        if [[ "${retries}" -lt 0 ]]; then
            return "${status}"
        fi

        sleep "${delay}"
    done
}
