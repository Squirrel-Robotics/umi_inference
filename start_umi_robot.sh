#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash start_umi_robot.sh [RATE_HZ]

This XR command takes no task or checkpoint argument. The generic deployment
contract is supplied by the policy server running on the 5090 machine.
The optional argument selects any positive action-waypoint playback rate in
Hz, including decimal values; the default is 10. A profile with an explicit
execution schedule (for example 30 Hz/H90 sampled at 10 Hz) requires the rate
to match that reviewed schedule. PI05_CHUNK_SIZE selects the planned switch
position in executable robot points; its default is 40. The next inference is
prefetched PI05_INFERENCE_LEAD_STEPS points before that position (default 10).
After the no-command policy prechecks pass, startup sends one direct all-zero
target to both dexterous hands without rate limiting, then confirms ACK and
near-zero feedback. Clear the space around both hands before running it.
This launcher does not run ~/pi05/zero_arms_z086.py; run that arm-zero script
manually first. For every accepted policy chunk, the first model action that
is actually dispatched starts a left-arm execution anchor 8 mm lower on
base_link Z. The same offset remains on every later action in that chunk, so
the model's Delta EEF differences are preserved and action 1 has no artificial
8 mm upward recovery. This is applied after decoding and before interpolation;
there is no extra waypoint and the recorded model action/anchor stays unchanged.
An asynchronous chunk that starts at k>0 applies it from that k onward.

PI05_ARM_COMMAND_RATE_HZ selects the interpolated dual-arm SDK rate; its
default is 50 Hz. It does not change the action/chunk playback speed. Set it
equal to RATE_HZ to disable intermediate arm points.
EOF
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "ERROR: the XR launcher accepts at most one control-rate argument." >&2
  echo "Usage: bash start_umi_robot.sh [RATE_HZ]" >&2
  exit 2
fi

root="${PI05_RUNTIME_ROOT:-/home/xr/pi05_umi_inference}"
runtime="${root}/runtime"
enable_file="${root}/LIVE_ENABLE"
authority_file="${root}/ARM_AUTHORITY_ACTIVE"
stop_generation_file="${root}/STOP_GENERATION"
stop_generation_lock="${root}/STOP_GENERATION.lock"
pid_file="${root}/umi_v2_robot.pid"
# Timestamped ROS camera1 and camera3 are the only wrist-camera sources.
camera_sync_max_skew_ms="50"
chunk_size="${PI05_CHUNK_SIZE:-40}"
inference_lead_steps="${PI05_INFERENCE_LEAD_STEPS:-10}"
control_rate_hz="${1:-${PI05_CONTROL_RATE_HZ:-10}}"
arm_command_rate_hz="${PI05_ARM_COMMAND_RATE_HZ:-50}"
action_basis_config="${PI05_V2_TO_CURRENT_ACTION_BASIS_CONFIG:-${root}/V2_TO_CURRENT_ACTION_BASIS.json}"
policy_address="${PI05_POLICY_ADDRESS:-192.168.110.199}"
policy_port="${UMI_POLICY_PORT:-8000}"
ros_setup="${PI05_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
python_bin="${PI05_PYTHON_BIN:-/home/xr/robocontrol_ws/.venv/bin/python}"
emergency_python="${PI05_EMERGENCY_PYTHON_BIN:-${python_bin}}"
emergency_helper="${PI05_EMERGENCY_STOP_HELPER:-${runtime}/x2_emergency_stop.py}"
python_pid=""
python_start_ticks=""
watchdog_pid=""
stop_requested=false
cleanup_done=false
emergency_failure=false

if [[ ! "${chunk_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PI05_CHUNK_SIZE must be a positive integer; got ${chunk_size}." >&2
  exit 2
fi
if [[ ! "${inference_lead_steps}" =~ ^[1-9][0-9]*$ ]] \
    || (( inference_lead_steps >= chunk_size )); then
  echo "ERROR: PI05_INFERENCE_LEAD_STEPS must satisfy 1 <= lead < chunk_size; got lead=${inference_lead_steps} chunk_size=${chunk_size}." >&2
  exit 2
fi

if [[ ! "${control_rate_hz}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v rate="${control_rate_hz}" 'BEGIN { exit !(rate > 0) }'; then
  echo "ERROR: control rate must be a positive finite number in Hz; got ${control_rate_hz}." >&2
  exit 2
fi
if [[ ! "${arm_command_rate_hz}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v rate="${arm_command_rate_hz}" 'BEGIN { exit !(rate > 0) }'; then
  echo "ERROR: PI05_ARM_COMMAND_RATE_HZ must be a positive finite number; got ${arm_command_rate_hz}." >&2
  exit 2
fi

read_stop_generation() {
  local generation=0
  flock -x 8
  if [[ -e "${stop_generation_file}" ]]; then
    IFS= read -r generation <"${stop_generation_file}" || generation=""
  fi
  flock -u 8
  if [[ ! "${generation}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf '%s\n' "${generation}"
}

check_stop_latch() {
  local stage="${1:-startup}"
  local current_generation
  if ! current_generation="$(read_stop_generation)"; then
    echo "CRITICAL: invalid STOP_GENERATION; startup is disabled." >&2
    exit 1
  fi
  if [[ "${current_generation}" != "${start_stop_generation}" ]]; then
    echo "STOP_LATCHED stage=${stage} start_generation=${start_stop_generation} current_generation=${current_generation}; startup will not enable robot output." >&2
    exit 130
  fi
}

run_external_emergency_stop() {
  local reason="${1:-stuck_live_controller}"
  # Bound import, SDK connection and RPC as one transaction. The helper's
  # --timeout only covers the RPC and cannot protect against a stuck connect.
  if timeout --kill-after=1s 3s \
      "${emergency_python}" "${emergency_helper}" \
      --timeout 1.0 --reason "${reason}"; then
    return 0
  fi
  echo "CRITICAL: independent X2 emergency stop was not confirmed." >&2
  return 1
}

controller_is_running() {
  [[ -n "${python_pid}" && -n "${python_start_ticks}" ]] || return 1
  kill -0 "${python_pid}" 2>/dev/null || return 1
  local process_state current_start_ticks
  read -r process_state current_start_ticks < <(
    awk '{print $3, $22}' "/proc/${python_pid}/stat" 2>/dev/null || true
  )
  [[ "${current_start_ticks}" == "${python_start_ticks}" \
      && "${process_state}" != "Z" ]]
}

find_remaining_live_controller() {
  local pid process_state command_line
  for pid in $(pgrep -u "$(id -u)" -f '[u]mi_ros_live_chunked.py' || true); do
    process_state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ "${process_state}" != "Z" ]] || continue
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${command_line}" == *"${runtime}/umi_ros_live_chunked.py"* ]]; then
      printf '%s\n' "${pid}"
      return 0
    fi
  done
  return 1
}

# XR enables an HTTP proxy from ~/.bashrc by default. websockets >= 15 honors
# that proxy automatically, so a private-LAN policy connection would otherwise
# be sent to 192.168.110.11:10808 and time out during the HTTP Upgrade. Keep the
# proxy for unrelated traffic, but always connect to the policy host directly.
direct_hosts="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
case ",${direct_hosts}," in
  *",${policy_address},"*) ;;
  *) direct_hosts="${direct_hosts:+${direct_hosts},}${policy_address}" ;;
esac
export NO_PROXY="${direct_hosts}"
export no_proxy="${direct_hosts}"

cleanup() {
  local original_status=$?
  local remaining_pid=""
  if [[ "${cleanup_done}" == true ]]; then
    return
  fi
  cleanup_done=true
  trap - EXIT
  trap '' INT TERM
  rm -f -- "${enable_file}"
  if [[ -n "${watchdog_pid}" ]]; then
    if ! wait "${watchdog_pid}"; then
      emergency_failure=true
    fi
    watchdog_pid=""
  fi
  if controller_is_running; then
    kill -TERM "${python_pid}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      controller_is_running || break
      sleep 0.1
    done
    if controller_is_running; then
      if ! run_external_emergency_stop "wrapper_cleanup_timeout"; then
        emergency_failure=true
      else
        rm -f -- "${authority_file}"
      fi
      kill -KILL "${python_pid}" 2>/dev/null || true
      for _ in $(seq 1 10); do
        controller_is_running || break
        sleep 0.05
      done
    fi
  fi
  if controller_is_running; then
    echo "CRITICAL: live controller is still active; feedback bridge was not restored." >&2
    exit 1
  fi
  if [[ -n "${python_pid}" ]]; then
    wait "${python_pid}" 2>/dev/null || true
  fi
  if [[ -e "${authority_file}" ]]; then
    echo "ARM_AUTHORITY_RECOVERY sentinel remains after controller exit." >&2
    if run_external_emergency_stop "controller_exit_without_stop_receipt"; then
      rm -f -- "${authority_file}"
    else
      emergency_failure=true
    fi
  fi
  remaining_pid="$(find_remaining_live_controller || true)"
  if [[ -n "${remaining_pid}" ]]; then
    echo "CRITICAL: live controller PID ${remaining_pid} remains active; feedback bridge was not restored." >&2
    exit 1
  fi
  rm -f -- "${pid_file}"
  if ! bash "${runtime}/revo2_bridge.sh" feedback; then
    echo "CRITICAL: failed to restore Revo2 feedback-only bridge." >&2
    exit 1
  fi
  if [[ "${emergency_failure}" == true ]]; then
    echo "CRITICAL: controller was stopped, but X2 emergency-stop confirmation failed." >&2
    exit 1
  fi
  echo "Stopped: no more arm/hand targets will be published."
  return "${original_status}"
}

request_stop() {
  local forwarded_signal="${1:-INT}"
  local signal_status=143
  if [[ "${forwarded_signal}" == "INT" ]]; then
    signal_status=130
  fi
  rm -f -- "${enable_file}"
  if [[ "${stop_requested}" == false ]]; then
    stop_requested=true
    echo "STOP_REQUESTED forwarding ${forwarded_signal} to live controller..."
    if controller_is_running; then
      kill -s "${forwarded_signal}" "${python_pid}" 2>/dev/null || true
      (
        trap - EXIT INT TERM
        sleep 2
        if controller_is_running; then
          echo "STOP_WATCHDOG escalating to TERM..." >&2
          kill -TERM "${python_pid}" 2>/dev/null || true
          sleep 1
        fi
        if controller_is_running; then
          echo "STOP_WATCHDOG controller did not exit; requesting independent X2 emergency stop." >&2
          emergency_status=0
          run_external_emergency_stop "ctrl_c_watchdog_timeout" \
            || emergency_status=$?
          if [[ "${emergency_status}" -eq 0 ]]; then
            rm -f -- "${authority_file}"
          fi
          echo "STOP_WATCHDOG forcing stuck controller process to exit." >&2
          kill -KILL "${python_pid}" 2>/dev/null || true
          exit "${emergency_status}"
        fi
      ) &
      watchdog_pid=$!
    else
      echo "STOP_REQUESTED before live child launch; aborting startup."
      exit "${signal_status}"
    fi
  else
    echo "SECOND_STOP_REQUEST escalating controller stop..." >&2
    if controller_is_running; then
      kill -s "${forwarded_signal}" "${python_pid}" 2>/dev/null || true
    fi
  fi
}

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required for single-controller ownership." >&2
  exit 1
fi
mkdir -p "${root}"
exec 8>"${stop_generation_lock}"
if ! start_stop_generation="$(read_stop_generation)"; then
  echo "CRITICAL: invalid STOP_GENERATION; startup is disabled." >&2
  exit 1
fi
exec 9>"${root}/umi_v2_robot.lock"
if ! flock -n 9; then
  echo "ERROR: another UMI v2 wrapper owns the controller lock." >&2
  exit 1
fi
if pgrep -f '[u]mi_ros_live_chunked.py' >/dev/null; then
  echo "ERROR: another UMI live controller is already running." >&2
  exit 1
fi
# start_stop_generation is the operator's start intent snapshot. Any stop that
# wins the generation lock afterwards increments the value; every following
# hardware-transition boundary then exits. A stop can never be consumed as an
# old latch merely because it raced with wrapper-lock acquisition.
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required for reliable Ctrl+C forwarding." >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "ERROR: timeout is required for bounded emergency-stop recovery." >&2
  exit 1
fi
if [[ -e "${authority_file}" ]]; then
  echo "ARM_AUTHORITY_RECOVERY stale sentinel found before startup." >&2
  if run_external_emergency_stop "stale_authority_before_start"; then
    rm -f -- "${authority_file}"
    echo "SAFETY_STOP_CONFIRMED: inspect/recover the robot, then run this command again." >&2
    exit 3
  else
    echo "CRITICAL: refusing startup while prior arm authority is unresolved." >&2
    exit 1
  fi
fi
check_stop_latch "before_policy_health"

# Do not switch the hand bridge or create LIVE_ENABLE until the policy server
# is accepting requests directly over the LAN.
policy_ready=false
for _ in $(seq 1 60); do
  check_stop_latch "policy_health_wait"
  if curl --noproxy '*' -fsS --max-time 2 \
      "http://${policy_address}:${policy_port}/healthz" >/dev/null; then
    policy_ready=true
    break
  fi
  sleep 1
done
check_stop_latch "after_policy_health"
if [[ "${policy_ready}" != true ]]; then
  echo "ERROR: policy server is not ready at ${policy_address}:${policy_port}." >&2
  exit 1
fi
echo "Policy server ready: ${policy_address}:${policy_port}"

# Resolve and validate the checkpoint-selected task/profile before changing the
# Revo2 bridge or creating LIVE_ENABLE. The controller repeats this validation
# on its own WebSocket connection before it can publish a command.
check_stop_latch "before_profile_preflight"
"${python_bin}" "${runtime}/policy_profile_preflight.py" \
  --host "${policy_address}" \
  --port "${policy_port}" \
  --requested-control-rate "${control_rate_hz}" \
  --chunk-size "${chunk_size}" \
  --v2-to-current-action-basis-config "${action_basis_config}"
check_stop_latch "after_profile_preflight"

# Camera readiness is established before any control authority changes. The
# bridge reuses a complete, fresh vendor pair; a partial vendor pair fails
# closed instead of introducing duplicate publishers.
check_stop_latch "before_wrist_camera_bridge"
bash "${runtime}/wrist_camera_bridge.sh" start
check_stop_latch "after_wrist_camera_bridge"

trap cleanup EXIT
trap 'request_stop INT' INT
trap 'request_stop TERM' TERM
check_stop_latch "before_control_bridge"
bash "${runtime}/revo2_bridge.sh" control
check_stop_latch "after_control_bridge"
touch "${enable_file}"
check_stop_latch "after_live_enable"
# ROS Jazzy's generated setup files probe optional AMENT variables that may be
# unset. Temporarily disable nounset only while loading the ROS environment.
set +u
source "${ros_setup}"
set -u
check_stop_latch "before_live_child_launch"

setsid "${python_bin}" \
  "${runtime}/umi_ros_live_chunked.py" \
  --host "${policy_address}" \
  --port "${policy_port}" \
  --rate "${control_rate_hz}" \
  --arm-command-rate "${arm_command_rate_hz}" \
  --chunk-size "${chunk_size}" \
  --inference-lead-steps "${inference_lead_steps}" \
  --camera-sync-max-skew-ms "${camera_sync_max_skew_ms}" \
  --v2-to-current-action-basis-config "${action_basis_config}" \
  --enable-file "${enable_file}" \
  --arm-authority-file "${authority_file}" &
python_pid=$!
if ! python_start_ticks="$(awk '{print $22}' "/proc/${python_pid}/stat" 2>/dev/null)" \
    || [[ -z "${python_start_ticks}" ]]; then
  set +e
  wait "${python_pid}" 2>/dev/null
  early_status=$?
  set -e
  if [[ "${early_status}" -eq 0 ]]; then
    early_status=1
  fi
  echo "ERROR: live controller exited before its process identity could be recorded (status=${early_status})." >&2
  exit "${early_status}"
fi
printf '%s %s\n' "${python_pid}" "${python_start_ticks}" >"${pid_file}"

# A signal interrupts bash's wait but the isolated Python process continues
# until request_stop forwards it. Keep waiting until that exact child is gone;
# only then may the EXIT cleanup restore the feedback-only Revo2 bridge.
python_status=0
set +e
while true; do
  wait "${python_pid}" 2>/dev/null
  wait_status=$?
  if ! controller_is_running; then
    python_status=${wait_status}
    break
  fi
done
set -e

if [[ -n "${watchdog_pid}" ]]; then
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
  watchdog_pid=""
fi
rm -f -- "${pid_file}"
python_pid=""
python_start_ticks=""
exit "${python_status}"
