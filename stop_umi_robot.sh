#!/usr/bin/env bash
set -euo pipefail

root="${PI05_RUNTIME_ROOT:-/home/xr/pi05_umi_inference}"
runtime="${root}/runtime"
enable_file="${root}/LIVE_ENABLE"
authority_file="${root}/ARM_AUTHORITY_ACTIVE"
stop_generation_file="${root}/STOP_GENERATION"
stop_generation_lock="${root}/STOP_GENERATION.lock"
pid_file="${root}/umi_v2_robot.pid"
int_grace_steps="${PI05_STOP_INT_GRACE_STEPS:-30}"
term_grace_steps="${PI05_STOP_TERM_GRACE_STEPS:-10}"
stop_poll_seconds="${PI05_STOP_POLL_SECONDS:-0.1}"
emergency_python="${PI05_EMERGENCY_PYTHON_BIN:-/home/xr/robocontrol_ws/.venv/bin/python}"
emergency_helper="${PI05_EMERGENCY_STOP_HELPER:-${runtime}/x2_emergency_stop.py}"
emergency_failure=false
# Once this stop transaction begins it must reach either confirmed controller
# death + bridge restore, or an explicit fail-closed error. Do not let a second
# terminal signal interrupt it between those two states.
trap '' INT TERM
# Remove the gate immediately, then publish a monotonic stop generation under
# flock. A concurrently starting wrapper snapshots/compares this generation at
# every transition and therefore cannot recreate a live gate after this stop.
rm -f -- "${enable_file}"
if ! command -v flock >/dev/null 2>&1; then
  echo "CRITICAL: flock is required for a race-free external stop." >&2
  exit 1
fi
mkdir -p "${root}"
exec 8>"${stop_generation_lock}"
flock -x 8
stop_generation=0
if [[ -e "${stop_generation_file}" ]]; then
  IFS= read -r stop_generation <"${stop_generation_file}" \
    || stop_generation=""
fi
if [[ ! "${stop_generation}" =~ ^[0-9]+$ ]]; then
  echo "WARNING: replacing invalid STOP_GENERATION during fail-safe stop." >&2
  stop_generation=0
fi
stop_generation=$((10#${stop_generation} + 1))
printf '%s\n' "${stop_generation}" >"${stop_generation_file}"
flock -u 8
rm -f -- "${enable_file}"
echo "STOP_LATCH_GENERATION=${stop_generation}"

run_external_emergency_stop() {
  local reason="${1:-external_stop_timeout}"
  if timeout --kill-after=1s 3s \
      "${emergency_python}" "${emergency_helper}" \
      --timeout 1.0 --reason "${reason}"; then
    return 0
  fi
  echo "CRITICAL: independent X2 emergency stop was not confirmed." >&2
  return 1
}

if ! command -v timeout >/dev/null 2>&1; then
  echo "CRITICAL: timeout is required for bounded emergency-stop recovery." >&2
  exit 1
fi

controller_pid=""
controller_start_ticks=""
if [[ -r "${pid_file}" ]]; then
  read -r candidate_pid candidate_start_ticks <"${pid_file}" || true
  if [[ "${candidate_pid:-}" =~ ^[0-9]+$ ]] \
      && [[ "${candidate_start_ticks:-}" =~ ^[0-9]+$ ]] \
      && kill -0 "${candidate_pid}" 2>/dev/null; then
    command_line="$(tr '\0' ' ' <"/proc/${candidate_pid}/cmdline" 2>/dev/null || true)"
    current_start_ticks="$(awk '{print $22}' "/proc/${candidate_pid}/stat" 2>/dev/null || true)"
    if [[ "${command_line}" == *"${runtime}/umi_ros_live_chunked.py"* ]] \
        && [[ "${current_start_ticks}" == "${candidate_start_ticks}" ]]; then
      controller_pid="${candidate_pid}"
      controller_start_ticks="${candidate_start_ticks}"
    else
      echo "WARNING: refusing to signal stale/unrelated PID ${candidate_pid}." >&2
    fi
  fi
fi

controller_is_running() {
  [[ -n "${controller_pid}" && -n "${controller_start_ticks}" ]] || return 1
  kill -0 "${controller_pid}" 2>/dev/null || return 1
  local process_state current_start_ticks
  read -r process_state current_start_ticks < <(
    awk '{print $3, $22}' "/proc/${controller_pid}/stat" 2>/dev/null || true
  )
  [[ "${current_start_ticks}" == "${controller_start_ticks}" \
      && "${process_state}" != "Z" ]]
}

find_live_controllers() {
  local pid process_state command_line
  # pgrep performs the /proc scan once in-process. Spawning stat/awk/tr for
  # every PID made the emergency stop path take seconds on a busy robot host.
  for pid in $(pgrep -u "$(id -u)" -f '[u]mi_ros_live_chunked.py' || true); do
    process_state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ "${process_state}" != "Z" ]] || continue
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${command_line}" == *"${runtime}/umi_ros_live_chunked.py"* ]]; then
      printf '%s\n' "${pid}"
    fi
  done
}

if [[ -z "${controller_pid}" ]]; then
  mapfile -t untracked_pids < <(find_live_controllers || true)
  if [[ "${#untracked_pids[@]}" -eq 1 ]]; then
    controller_pid="${untracked_pids[0]}"
    controller_start_ticks="$(
      awk '{print $22}' "/proc/${controller_pid}/stat" 2>/dev/null || true
    )"
    if [[ -z "${controller_start_ticks}" ]]; then
      echo "CRITICAL: could not bind untracked controller PID ${controller_pid} to its process identity." >&2
      exit 1
    fi
    echo "WARNING: adopting exact live controller PID ${controller_pid} because its PID record is missing/stale." >&2
  elif [[ "${#untracked_pids[@]}" -gt 1 ]]; then
    echo "CRITICAL: multiple untracked live controllers are active: ${untracked_pids[*]}." >&2
    if run_external_emergency_stop "multiple_untracked_live_controllers"; then
      rm -f -- "${authority_file}"
    fi
    echo "CRITICAL: processes were not signalled because controller ownership is ambiguous; feedback bridge was not restored." >&2
    exit 1
  fi
fi

if [[ -n "${controller_pid}" ]]; then
  echo "STOP_REQUESTED pid=${controller_pid}"
  kill -INT "${controller_pid}" 2>/dev/null || true
  for _ in $(seq 1 "${int_grace_steps}"); do
    controller_is_running || break
    sleep "${stop_poll_seconds}"
  done
  if controller_is_running; then
    echo "STOP_WATCHDOG escalating to TERM..." >&2
    kill -TERM "${controller_pid}" 2>/dev/null || true
    for _ in $(seq 1 "${term_grace_steps}"); do
      controller_is_running || break
      sleep "${stop_poll_seconds}"
    done
  fi
  if controller_is_running; then
    echo "STOP_WATCHDOG controller is stuck; requesting independent X2 emergency stop." >&2
    if ! run_external_emergency_stop; then
      emergency_failure=true
    else
      rm -f -- "${authority_file}"
    fi
    echo "STOP_WATCHDOG forcing stuck controller process to exit." >&2
    kill -KILL "${controller_pid}" 2>/dev/null || true
    for _ in $(seq 1 "${term_grace_steps}"); do
      controller_is_running || break
      sleep "${stop_poll_seconds}"
    done
  fi
  if controller_is_running; then
    echo "CRITICAL: controller PID ${controller_pid} is still active; feedback bridge was not restored." >&2
    exit 1
  fi
fi

remaining_pid="$(find_live_controllers | head -n 1 || true)"
if [[ -n "${remaining_pid}" ]]; then
  echo "CRITICAL: another live controller PID ${remaining_pid} remains active; feedback bridge was not restored." >&2
  exit 1
fi

# A controller may crash after taking arm authority but before its in-process
# hold/stop receipt is written. In that case the process is already gone, so
# the sentinel is the only durable evidence that an independent stop is due.
if [[ -e "${authority_file}" ]]; then
  echo "ARM_AUTHORITY_RECOVERY sentinel remains after controller exit." >&2
  if run_external_emergency_stop; then
    rm -f -- "${authority_file}"
  else
    emergency_failure=true
  fi
fi

rm -f -- "${pid_file}"
# This is safe only after the exact live child has exited (or when no verified
# child exists). The foreground wrapper performs the same idempotent cleanup.
if ! bash "${runtime}/revo2_bridge.sh" feedback; then
  echo "CRITICAL: failed to restore Revo2 feedback-only bridge." >&2
  exit 1
fi
if [[ "${emergency_failure}" == true ]]; then
  echo "CRITICAL: controller was stopped, but X2 emergency-stop confirmation failed." >&2
  exit 1
fi
echo "pi05 publishing stopped; live commands are disabled."
