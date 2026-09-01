#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash start_umi_server.sh CHECKPOINT {full|roi}

Examples:
  bash start_umi_server.sh /mnt/dzq/checkpoint/umi_task_v1/6000 full
  bash start_umi_server.sh /mnt/dzq/checkpoint/umi_task_v1_30hz/4000 roi
  bash start_umi_server.sh /mnt/dzq/checkpoint/umi_task_v1_new_10hz_full/1000 full
  UMI_CHECKPOINT=/mnt/dzq/checkpoint/umi_task_v1/6000 \
    UMI_CAM_HIGH_MODE=full bash start_umi_server.sh

No deployment_profile.json or checkpoint/profile/hash allowlist is used.
The checkpoint must restore successfully and warm up to finite actions=(50,30).
Choose full or roi explicitly because camera preprocessing is not stored in
Orbax. The fixed deployment mapping is identity + shared anchor + right multiply.
Set UMI_TASK_PROMPT to override the default
'Put the object on the box ,then return it back.'.
Set E6_COMPONENT=package/activity only when deploying a different E6 app build.
E6_TIMESTAMP_OFFSET_MS is optional and means (5090 realtime - E6 realtime).
Leave it unset unless that cross-device clock offset has been measured.
EOF
  exit 0
fi

checkpoint=""
cam_high_mode=""
if [[ -n "${UMI_CHECKPOINT:-}" ]]; then
  if [[ $# -ne 0 ]]; then
    echo "ERROR: choose positional CHECKPOINT or UMI_CHECKPOINT, not both." >&2
    exit 2
  fi
  checkpoint="${UMI_CHECKPOINT}"
  cam_high_mode="${UMI_CAM_HIGH_MODE:-}"
else
  if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "ERROR: checkpoint and explicit cam_high mode are required." >&2
    echo "Usage: bash start_umi_server.sh CHECKPOINT {full|roi}" >&2
    exit 2
  fi
  checkpoint="$1"
  if [[ $# -eq 2 ]]; then
    if [[ -n "${UMI_CAM_HIGH_MODE:-}" ]]; then
      echo "ERROR: choose positional cam_high mode or UMI_CAM_HIGH_MODE, not both." >&2
      exit 2
    fi
    cam_high_mode="$2"
  else
    cam_high_mode="${UMI_CAM_HIGH_MODE:-}"
  fi
fi
case "${cam_high_mode}" in
  roi|full) ;;
  *)
    echo "ERROR: generic deployment requires explicit cam_high mode: full or roi." >&2
    exit 2
    ;;
esac
if [[ -z "${checkpoint}" ]]; then
  echo "ERROR: checkpoint must not be empty." >&2
  exit 2
fi
if ! checkpoint="$(realpath -e -- "${checkpoint}")"; then
  echo "ERROR: checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi
e6_component="${E6_COMPONENT:-com.ssnwt.e6stream.debug/com.ssnwt.e6stream.VrNativeActivity}"
e6_forward_port="${E6_FORWARD_PORT:-18554}"
policy_port="${UMI_POLICY_PORT:-8000}"
record_root="${UMI_RECORD_ROOT:-/home/dzq/pi05_inference_records}"
record_min_free_gb="${UMI_RECORD_MIN_FREE_GB:-20}"
camera_sync_max_skew_ms="${PI05_CAMERA_SYNC_MAX_SKEW_MS:-50}"
task_prompt="${UMI_TASK_PROMPT:-Put the object on the box ,then return it back.}"
if [[ -z "${task_prompt//[[:space:]]/}" ]]; then
  echo "ERROR: UMI_TASK_PROMPT must not be empty." >&2
  exit 2
fi
e6_history_seconds="${E6_HISTORY_SECONDS:-8}"
e6_timestamp_offset_ms="${E6_TIMESTAMP_OFFSET_MS:-}"
e6_timestamp_args=()
if [[ -n "${e6_timestamp_offset_ms}" ]]; then
  e6_timestamp_args=(--e6-timestamp-offset-ms "${e6_timestamp_offset_ms}")
fi
server_lock="/home/dzq/pi05_inference/umi_v2_server.lock"

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required for single-server ownership." >&2
  exit 1
fi
exec 9>"${server_lock}"
if ! flock -n 9; then
  echo "ERROR: another UMI server launcher or install owns the policy lock." >&2
  exit 1
fi

# The adb client can start a persistent adb server. Close the policy-lock file
# descriptor only in adb children so that daemon cannot keep the lock after the
# foreground policy process exits. The launcher and final policy process retain
# fd 9, so single-server ownership still covers the full policy lifetime.
adb_without_policy_lock() (
  exec 9>&-
  exec adb "$@"
)

if systemctl --user is-active --quiet umi-pi05.service 2>/dev/null; then
  echo "ERROR: old umi-pi05.service is still active. Stop it before using this foreground flow." >&2
  exit 1
fi
if ss -ltnH "sport = :${policy_port}" | grep -q .; then
  echo "ERROR: TCP port ${policy_port} is already occupied. Stop the old policy process first." >&2
  exit 1
fi
test -f "${checkpoint}/_CHECKPOINT_METADATA"
test -f "${checkpoint}/params/_METADATA"
test -d "${checkpoint}/assets"
echo "CHECKPOINT_SELECTED path=${checkpoint}"

# Validate layout, 30-D norm stats and the explicit full/ROI choice before ADB
# wakes E6. No manifest, asset, norm-hash or weights allowlist is consulted.
PYTHONPATH="/home/dzq/pi05_inference${PYTHONPATH:+:${PYTHONPATH}}" \
  /home/dzq/openpi/.venv/bin/python \
  /home/dzq/pi05_inference/umi_live_contract.py \
  --check-cam-high-mode "${cam_high_mode}" \
  --generic-prompt "${task_prompt}" \
  "${checkpoint}"

adb_without_policy_lock wait-for-device
if ! resolved_e6_component="$(
  adb_without_policy_lock shell cmd package resolve-activity --brief "${e6_component}" 2>&1 \
    | tr -d '\r' \
    | tail -n 1
)"; then
  echo "ERROR: failed to resolve E6 activity: ${e6_component}" >&2
  exit 1
fi
if [[ "${resolved_e6_component}" != "${e6_component}" ]]; then
  echo "ERROR: E6 activity is not installed or does not resolve exactly." >&2
  echo "  requested=${e6_component}" >&2
  echo "  resolved=${resolved_e6_component}" >&2
  exit 1
fi
echo "E6_COMPONENT_READY component=${resolved_e6_component}"
adb_without_policy_lock shell input keyevent KEYCODE_WAKEUP >/dev/null
adb_without_policy_lock shell am start -n "${e6_component}" >/dev/null
adb_without_policy_lock forward "tcp:${e6_forward_port}" tcp:8554 >/dev/null

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME=/home/dzq/openpi-assets

cd /home/dzq/openpi
exec /home/dzq/openpi/.venv/bin/python \
  /home/dzq/pi05_inference/serve_umi_v2_policy.py \
  --checkpoint "${checkpoint}" \
  --host 0.0.0.0 \
  --port "${policy_port}" \
  --e6-host 127.0.0.1 \
  --e6-port "${e6_forward_port}" \
  --e6-eye right \
  --cam-high-mode "${cam_high_mode}" \
  --task-prompt "${task_prompt}" \
  --e6-max-age-ms "${E6_MAX_AGE_MS:-100}" \
  --e6-history-seconds "${e6_history_seconds}" \
  "${e6_timestamp_args[@]}" \
  --camera-sync-max-skew-ms "${camera_sync_max_skew_ms}" \
  --record-root "${record_root}" \
  --record-min-free-gb "${record_min_free_gb}"
