#!/usr/bin/env bash
set -euo pipefail

action="${1:-start}"
[[ "${action}" == "start" ]] || { echo "Usage: bash wrist_camera_bridge.sh start" >&2; exit 2; }

root="${PI05_RUNTIME_ROOT:-/home/xr/pi05_umi_inference}"
runtime="${root}/runtime"
ros_setup="${PI05_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
python_bin="${PI05_PYTHON_BIN:-/home/xr/robocontrol_ws/.venv/bin/python}"
unit="pi05-wrist-cameras.service"
left="/camera1/usb_cam1/image_raw/image_compressed"
right="/camera3/usb_cam3/image_raw/image_compressed"

set +u
source "${ros_setup}"
set -u

publisher_count() {
  ros2 topic info "$1" 2>/dev/null | awk '/Publisher count:/ {print $3; found=1} END {if (!found) print 0}'
}

has_frame() {
  timeout 3 ros2 topic echo --once "$1" sensor_msgs/msg/CompressedImage \
    --field header.stamp >/dev/null 2>&1
}

left_count="$(publisher_count "${left}")"
right_count="$(publisher_count "${right}")"
if (( left_count > 0 && right_count > 0 )); then
  if has_frame "${left}" && has_frame "${right}"; then
    echo "Wrist cameras ready: reusing existing vendor publishers."
    exit 0
  fi
  echo "ERROR: existing wrist-camera publishers are not delivering two fresh frames; refusing duplicates." >&2
  exit 1
fi
if (( left_count > 0 || right_count > 0 )); then
  echo "ERROR: only one wrist-camera topic has a publisher (left=${left_count}, right=${right_count}); refusing duplicate publishers." >&2
  exit 1
fi

systemctl --user reset-failed "${unit}" >/dev/null 2>&1 || true
systemd-run --user --unit="${unit%.service}" --collect \
  --property=Restart=on-failure --property=RestartSec=1 \
  --property=KillMode=mixed \
  /usr/bin/bash -lc \
  "set +u; source '${ros_setup}'; set -u; exec '${python_bin}' '${runtime}/wrist_camera_publisher.py'"

ready=false
for _ in $(seq 1 15); do
  left_count="$(publisher_count "${left}")"
  right_count="$(publisher_count "${right}")"
  if (( left_count == 1 && right_count == 1 )) \
      && has_frame "${left}" && has_frame "${right}"; then
    ready=true
    break
  fi
  if ! systemctl --user is-active --quiet "${unit}"; then
    break
  fi
  sleep 1
done
if [[ "${ready}" == true ]]; then
  echo "Wrist cameras ready: independent publisher is active."
  exit 0
fi

echo "ERROR: independent wrist-camera publisher did not produce both real frames." >&2
systemctl --user status --no-pager "${unit}" >&2 || true
journalctl --user -u "${unit}" -n 80 --no-pager >&2 || true
systemctl --user stop "${unit}" >/dev/null 2>&1 || true
exit 1
