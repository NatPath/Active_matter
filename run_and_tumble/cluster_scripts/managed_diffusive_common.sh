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

managed_normalize_case() {
    local case_name="$1"
    case "${case_name}" in
        diffusive_1d_pmlr|pmlr|PmLr)
            printf "diffusive_1d_pmlr"
            ;;
        single_origin_bond|origin_bond|center_bond|centred_bond|centered_bond)
            printf "single_origin_bond"
            ;;
        *)
            echo "Unsupported managed case '${case_name}'. Use diffusive_1d_pmlr or single_origin_bond." >&2
            return 1
            ;;
    esac
}

managed_case_family() {
    local case_name
    case_name="$(managed_normalize_case "$1")" || return 1
    printf "%s" "${case_name}"
}

managed_case_label() {
    local case_name
    case_name="$(managed_normalize_case "$1")" || return 1
    case "${case_name}" in
        diffusive_1d_pmlr) printf "diffusive 1D PmLr" ;;
        single_origin_bond) printf "single origin bond" ;;
    esac
}

managed_default_run_id() {
    local case_name="$1"
    local L="$2"
    local rho="$3"
    local arg1="$4"
    local arg2="$5"
    case_name="$(managed_normalize_case "${case_name}")" || return 1
    local rho_slug arg1_slug arg2_slug
    rho_slug="$(managed_slugify "$(managed_format_float "${rho}")")"
    arg1_slug="$(managed_slugify "$(managed_format_float "${arg1}")")"
    arg2_slug="$(managed_slugify "$(managed_format_float "${arg2}")")"
    case "${case_name}" in
        diffusive_1d_pmlr)
            printf "diffusive_1d_pmlr_L%s_rho%s_gamma%s_V%s" "${L}" "${rho_slug}" "${arg1_slug}" "${arg2_slug}"
            ;;
        single_origin_bond)
            printf "single_origin_bond_L%s_rho%s_f%s_ffr%s" "${L}" "${rho_slug}" "${arg1_slug}" "${arg2_slug}"
            ;;
    esac
}

managed_run_root() {
    local repo_root="$1"
    local case_name="$2"
    local run_id="$3"
    local family
    family="$(managed_case_family "${case_name}")" || return 1
    printf "%s/runs/%s/managed/%s" "${repo_root}" "${family}" "${run_id}"
}

managed_registry_file() {
    local repo_root="$1"
    local case_name="$2"
    local family
    family="$(managed_case_family "${case_name}")" || return 1
    case "${family}" in
        diffusive_1d_pmlr)
            printf "%s/runs/%s/run_registry.csv" "${repo_root}" "${family}"
            ;;
        single_origin_bond)
            printf "%s/runs/%s/managed_registry.csv" "${repo_root}" "${family}"
            ;;
    esac
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

managed_require_nonnegative_float() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] ||
       ! awk -v x="${value}" 'BEGIN { exit !(x >= 0.0) }'; then
        echo "--${name} must be a non-negative number. Got '${value}'." >&2
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
