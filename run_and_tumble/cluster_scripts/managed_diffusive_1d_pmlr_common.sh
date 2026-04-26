#!/usr/bin/env bash

managed_repo_root() {
    local script_dir="$1"
    if [[ -f "${script_dir}/../run_diffusive_no_activity.jl" ]]; then
        cd "${script_dir}/.." && pwd
    elif [[ -f "${script_dir}/run_diffusive_no_activity.jl" ]]; then
        cd "${script_dir}" && pwd
    else
        echo "Could not locate repo root from script location: ${script_dir}" >&2
        return 1
    fi
}

managed_slugify() {
    printf "%s" "$1" | sed -E 's/[+]/p/g; s/-/m/g; s/[.]/p/g; s/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//'
}

managed_format_float() {
    awk -v x="$1" 'BEGIN { printf "%.12g", x }'
}

managed_default_run_id() {
    local L="$1"
    local rho="$2"
    local gamma="$3"
    local potential_strength="$4"
    local rho_slug gamma_slug potential_slug
    rho_slug="$(managed_slugify "$(managed_format_float "${rho}")")"
    gamma_slug="$(managed_slugify "$(managed_format_float "${gamma}")")"
    potential_slug="$(managed_slugify "$(managed_format_float "${potential_strength}")")"
    printf "diffusive_1d_pmlr_L%s_rho%s_gamma%s_V%s" "${L}" "${rho_slug}" "${gamma_slug}" "${potential_slug}"
}

managed_run_root() {
    local repo_root="$1"
    local run_id="$2"
    printf "%s/runs/diffusive_1d_pmlr/managed/%s" "${repo_root}" "${run_id}"
}

managed_timestamp() {
    date +%Y%m%d-%H%M%S
}

managed_require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "--${name} must be a positive integer. Got '${value}'." >&2
        return 1
    fi
}

managed_require_positive_float() {
    local name="$1"
    local value="$2"
    if ! awk -v x="${value}" 'BEGIN { exit !(x > 0.0) }'; then
        echo "--${name} must be a positive number. Got '${value}'." >&2
        return 1
    fi
}

managed_yaml_value() {
    local path="$1"
    local key="$2"
    awk -v key="${key}" '
        $0 ~ "^[[:space:]]*" key ":" {
            value=$0
            sub("^[[:space:]]*" key ":[[:space:]]*", "", value)
            gsub(/^"/, "", value)
            gsub(/"$/, "", value)
            print value
            exit
        }
    ' "${path}"
}

managed_next_replica_index() {
    local replicas_csv="$1"
    if [[ ! -f "${replicas_csv}" ]]; then
        printf "1"
        return 0
    fi
    awk -F, '
        NR > 1 {
            id=$1
            if (id ~ /^r[0-9]+$/) {
                sub(/^r/, "", id)
                value=id + 0
                if (value > max) max=value
            }
        }
        END { print max + 1 }
    ' "${replicas_csv}"
}

managed_replica_id() {
    local idx="$1"
    printf "r%06d" "${idx}"
}
