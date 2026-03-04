#!/usr/bin/env bash

two_force_d_normalize_spacing_mode() {
    local mode="${1:-linear}"
    case "${mode}" in
        linear)
            echo "linear"
            ;;
        log_midpoints|log_mid|logmid|lm)
            echo "log_midpoints"
            ;;
        *)
            return 1
            ;;
    esac
}

two_force_d_spacing_tag() {
    local mode
    mode="$(two_force_d_normalize_spacing_mode "${1:-linear}")" || return 1
    local d_step="${2:-2}"
    if [[ "${mode}" == "log_midpoints" ]]; then
        echo "lm"
    else
        echo "s${d_step}"
    fi
}

two_force_d_generate_d_values() {
    local mode
    mode="$(two_force_d_normalize_spacing_mode "${1:-linear}")" || return 1
    local d_min="${2:-2}"
    local d_max="${3:-0}"
    local d_step="${4:-2}"

    if ! [[ "${d_min}" =~ ^[0-9]+$ && "${d_max}" =~ ^[0-9]+$ && "${d_step}" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    if (( d_max < d_min )); then
        return 0
    fi

    local -a values=()
    local d
    if [[ "${mode}" == "linear" ]]; then
        if (( d_step <= 0 )); then
            return 1
        fi
        for ((d = d_min; d <= d_max; d += d_step)); do
            values+=("${d}")
        done
    else
        # Log + midpoint spacing:
        #   powers of 2:        2,4,8,16,...
        #   midpoint branch:    3,6,12,24,... = 3*2^(k-1)
        #   then odd values are filtered out globally below.
        local p=2
        local midpoint
        while (( p <= d_max )); do
            values+=("${p}")
            midpoint=$(( (3 * p) / 2 ))
            if (( midpoint <= d_max )); then
                values+=("${midpoint}")
            fi
            p=$((p * 2))
        done
        if (( d_min > 2 )); then
            local -a filtered=()
            for d in "${values[@]}"; do
                if (( d >= d_min )); then
                    filtered+=("${d}")
                fi
            done
            values=("${filtered[@]}")
        fi
    fi

    # Enforce even d only (odd d breaks exact site-centered symmetry).
    local -a even_values=()
    for d in "${values[@]}"; do
        if (( d % 2 == 0 )); then
            even_values+=("${d}")
        fi
    done
    values=("${even_values[@]}")

    local last=""
    for d in "${values[@]}"; do
        if [[ -z "${last}" || "${d}" != "${last}" ]]; then
            echo "${d}"
            last="${d}"
        fi
    done
}

two_force_d_csv_to_array() {
    local csv="$1"
    local -n out_ref="$2"
    out_ref=()
    [[ -z "${csv}" ]] && return 0
    local item
    IFS=',' read -r -a out_ref <<< "${csv}"
    local -a filtered=()
    for item in "${out_ref[@]}"; do
        if ! [[ "${item}" =~ ^[0-9]+$ ]]; then
            return 1
        fi
        if (( item % 2 == 0 )); then
            filtered+=("${item}")
        fi
    done
    out_ref=("${filtered[@]}")
    return 0
}
