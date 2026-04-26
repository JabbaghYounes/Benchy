#!/bin/bash
# Shared library for the hardware-verification runners
# (verify_ai_hat_plus.sh, verify_ai_hat_plus_2.sh).
#
# Source this file from the entrypoints; it relies on scripts/common.sh
# for the colour + logging helpers (info, success, warn, error).
#
# Public surface
# ----------------
#   hw_init                       Set up results directory, counters, log dir.
#   hw_preflight_rpi_ai_hat_plus  Assert platform == AI HAT+ (Hailo-8/8L).
#   hw_preflight_rpi_ai_hat_plus_2  Assert platform == AI HAT+ 2 (Hailo-10H).
#   hw_run_step NAME CMD [validator-args…]
#                                 Run CMD, time it, capture stdout+stderr to
#                                 a log file, optionally validate the
#                                 produced bench_*.json against contracts.
#   hw_skip NAME REASON           Mark a step as skipped (counted separately).
#   hw_summary                    Print pass/fail/skip counts + total wall
#                                 time. Exits 0 on all-pass, 1 on any FAIL.
#
# Continue-on-failure semantics: each hw_run_step that fails is recorded
# but execution proceeds. The exit code is decided in hw_summary at the
# end so the operator sees the full sweep in one session.

# Locate scripts/common.sh relative to THIS file (hw_verify_common.sh),
# not relative to the entrypoint that sources us — the entrypoints are
# expected to live in the same scripts/ directory.
_HW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$_HW_SCRIPT_DIR/common.sh"

_HW_PROJECT_ROOT="$(cd "$_HW_SCRIPT_DIR/.." && pwd)"
_HW_VALIDATOR="$_HW_SCRIPT_DIR/hw_verify_validators.py"

# State -----------------------------------------------------------------------
HW_RESULTS_DIR=""
HW_LOGS_DIR=""
HW_START_TIME=0

# Per-step accounting. Bash arrays of "NAME|DURATION_S" entries.
HW_PASSED=()
HW_FAILED=()
HW_SKIPPED=()
# Index of the step we're about to run (1-based). Computed live by
# hw_run_step from the sum of the three arrays plus 1.
HW_TOTAL_STEPS=0   # set by entrypoints if they want a [i/N] prefix

# ----------------------------------------------------------------------------

hw_init() {
    # Initialise state: results + logs directory under results/, timestamped.
    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    HW_RESULTS_DIR="${HW_RESULTS_DIR:-$_HW_PROJECT_ROOT/results/hw_verify_$timestamp}"
    HW_LOGS_DIR="$HW_RESULTS_DIR/logs"
    mkdir -p "$HW_LOGS_DIR"
    HW_START_TIME=$(date +%s)

    info "Hardware-verification runner starting"
    info "Results bundle: $HW_RESULTS_DIR"
    info "Per-step logs : $HW_LOGS_DIR"
    echo
}

# Preflight gates -------------------------------------------------------------

_hw_assert_platform() {
    # Internal: confirm `python -m benchmark info` reports the expected
    # platform string. Avoids running an AI HAT+ 2 sweep on a Jetson by
    # accident. Skips the check if the venv isn't activated yet — the
    # entrypoint is expected to abort earlier in that case.
    local expected="$1"
    if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
        error "No python on PATH; activate the venv before running this script."
        exit 2
    fi
    local py
    py="$(command -v python || command -v python3)"
    local detected
    detected="$("$py" -m benchmark info 2>/dev/null \
        | awk -F: '/^[[:space:]]*platform:/ {print $2}' | tr -d ' ' | head -n1)"
    if [[ -z "$detected" ]]; then
        warn "Could not determine platform via 'python -m benchmark info'; continuing without preflight gate."
        return 0
    fi
    if [[ "$detected" != "$expected" ]]; then
        error "Platform mismatch: detected $detected, expected $expected."
        error "Run the matching script for your hardware, or pass --platform to override."
        exit 2
    fi
    success "Platform check OK ($detected)"
}

hw_preflight_rpi_ai_hat_plus() {
    info "Preflight: Pi 5 + AI HAT+ (Hailo-8 / 8L)"
    _hw_assert_platform "rpi_ai_hat_plus"
}

hw_preflight_rpi_ai_hat_plus_2() {
    info "Preflight: Pi 5 + AI HAT+ 2 (Hailo-10H)"
    _hw_assert_platform "rpi_ai_hat_plus_2"
}

# Step execution --------------------------------------------------------------

# hw_run_step NAME CMD [--workload WORKLOAD --task TASK --backend BE --require-npu-metrics]
#
# Runs CMD via `bash -c` so the caller can pass shell-style command lines.
# Captures stdout+stderr to "$HW_LOGS_DIR/<safe-name>.log". After CMD
# returns, if any --workload arg is present, invokes
# scripts/hw_verify_validators.py against $HW_RESULTS_DIR to assert
# contracts on the freshly-written bench_*.json. Validation failure
# demotes a green-exit run to FAIL.
hw_run_step() {
    local name="$1"
    local cmd="$2"
    shift 2
    local validator_args=("$@")

    # Compose a log-safe filename from NAME (strip "[experimental]" tag,
    # collapse spaces/colons to underscores).
    local safe_name
    safe_name="$(echo "$name" \
        | sed -e 's/\[experimental\]//g' \
              -e 's/[[:space:]]\+/_/g' \
              -e 's/[^A-Za-z0-9_.-]/_/g' \
              -e 's/__*/_/g' -e 's/_$//')"
    local log_path="$HW_LOGS_DIR/${safe_name}.log"

    local idx=$(( ${#HW_PASSED[@]} + ${#HW_FAILED[@]} + ${#HW_SKIPPED[@]} + 1 ))
    local prefix
    if (( HW_TOTAL_STEPS > 0 )); then
        prefix="$(printf '[%2d/%d]' "$idx" "$HW_TOTAL_STEPS")"
    else
        prefix="$(printf '[%2d]' "$idx")"
    fi

    info "${prefix} ${name} ..."
    local step_start
    step_start=$(date +%s)
    local rc
    if bash -c "$cmd" >"$log_path" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    local step_end
    step_end=$(date +%s)
    local duration=$(( step_end - step_start ))

    # Validation pass — only if the command itself succeeded AND the
    # caller asked for validation (any --workload arg).
    local validator_rc=0
    if (( rc == 0 )) && (( ${#validator_args[@]} > 0 )); then
        if ! python3 "$_HW_VALIDATOR" "$HW_RESULTS_DIR" "${validator_args[@]}" \
                >>"$log_path" 2>&1; then
            validator_rc=1
        fi
    fi

    if (( rc == 0 )) && (( validator_rc == 0 )); then
        success "${prefix} ${name} -- PASS (${duration}s)"
        HW_PASSED+=("${name}|${duration}")
    else
        local reason="exit=$rc"
        if (( validator_rc != 0 )); then
            reason="$reason validator=fail"
        fi
        error "${prefix} ${name} -- FAIL (${duration}s, ${reason})"
        error "         see ${log_path#$_HW_PROJECT_ROOT/}"
        HW_FAILED+=("${name}|${duration}")
    fi
}

hw_skip() {
    local name="$1"
    local reason="${2:-}"
    local idx=$(( ${#HW_PASSED[@]} + ${#HW_FAILED[@]} + ${#HW_SKIPPED[@]} + 1 ))
    local prefix
    if (( HW_TOTAL_STEPS > 0 )); then
        prefix="$(printf '[%2d/%d]' "$idx" "$HW_TOTAL_STEPS")"
    else
        prefix="$(printf '[%2d]' "$idx")"
    fi
    warn "${prefix} ${name} -- SKIP${reason:+ (${reason})}"
    HW_SKIPPED+=("${name}|0")
}

# Optional: trigger `python -m benchmark report` so the dashboard is
# ready immediately. Used only by the AI HAT+ 2 entrypoint where we
# want the cross-backend comparison artefact in the same session.
hw_finalize_with_report() {
    info "Generating dashboard from $HW_RESULTS_DIR ..."
    local py
    py="$(command -v python || command -v python3)"
    if "$py" -m benchmark report --input "$HW_RESULTS_DIR" \
            --output "$HW_RESULTS_DIR/report" >"$HW_LOGS_DIR/report.log" 2>&1; then
        success "Dashboard: $HW_RESULTS_DIR/report/*_dashboard.html"
    else
        warn "Dashboard generation failed; see $HW_LOGS_DIR/report.log"
    fi
}

# Summary ---------------------------------------------------------------------

hw_summary() {
    local total_time
    total_time=$(( $(date +%s) - HW_START_TIME ))
    local minutes=$(( total_time / 60 ))
    local seconds=$(( total_time % 60 ))

    echo
    echo "============================================================"
    echo "  Hardware verification complete"
    echo "============================================================"
    printf "  Passed:  %d\n" "${#HW_PASSED[@]}"
    printf "  Failed:  %d\n" "${#HW_FAILED[@]}"
    printf "  Skipped: %d\n" "${#HW_SKIPPED[@]}"
    printf "  Wall time: %dm %02ds\n" "$minutes" "$seconds"
    echo

    # Count v26 / experimental entries separately so a v26 failure
    # doesn't look like a project regression.
    local fail_total="${#HW_FAILED[@]}"
    local fail_experimental=0
    if (( fail_total > 0 )); then
        echo "Failures:"
        local entry name dur
        for entry in "${HW_FAILED[@]}"; do
            name="${entry%%|*}"
            dur="${entry##*|}"
            if [[ "$name" == *"[experimental]"* ]]; then
                fail_experimental=$(( fail_experimental + 1 ))
            fi
            printf "  - %s (%ss)\n" "$name" "$dur"
        done
        echo
    fi

    echo "Results bundle: $HW_RESULTS_DIR"
    echo

    local fail_blocking=$(( fail_total - fail_experimental ))
    if (( fail_blocking > 0 )); then
        error "${fail_blocking} blocking failure(s); see logs for details."
        return 1
    fi
    if (( fail_experimental > 0 )); then
        warn "${fail_experimental} experimental failure(s); v26 is still tagged experimental."
    fi
    success "All blocking checks passed."
    return 0
}
