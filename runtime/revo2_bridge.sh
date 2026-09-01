#!/usr/bin/env bash
set -euo pipefail

feedback_name="pi05-revo2-feedback"
control_name="pi05-revo2-control"
image="zbl-registry.cn-shenzhen.cr.aliyuncs.com/xr/runtime/cx002:core_v00.28.02"
root="${PI05_RUNTIME_ROOT:-/home/xr/pi05_umi_inference}"
feedback_script="${root}/revo2_feedback_only.py"
ros_setup="${PI05_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
left_port="${PI05_REVO2_LEFT_PORT:-/dev/serial/by-path/pci-0000:c7:00.3-usb-0:1.1.1:1.0-port0}"
right_port="${PI05_REVO2_RIGHT_PORT:-/dev/serial/by-path/pci-0000:c5:00.3-usb-0:3.1:1.0-port0}"
lidar_device="${PI05_LIDAR_DEVICE:-/dev/usb_lidar}"

usage() {
  echo "Usage: bash revo2_bridge.sh {control|feedback}" >&2
}

verify_serial_ownership() {
  local left_real right_real lidar_real="absent"
  if ! left_real="$(realpath -e -- "${left_port}")"; then
    echo "ERROR: left Revo2 serial device is missing: ${left_port}" >&2
    return 1
  fi
  if ! right_real="$(realpath -e -- "${right_port}")"; then
    echo "ERROR: right Revo2 serial device is missing: ${right_port}" >&2
    return 1
  fi
  if [[ ! -c "${left_real}" || ! -c "${right_real}" ]]; then
    echo "ERROR: Revo2 paths must resolve to character devices." >&2
    return 1
  fi
  if [[ "${left_real}" == "${right_real}" ]]; then
    echo "ERROR: left and right Revo2 paths resolve to the same device: ${left_real}" >&2
    return 1
  fi

  if [[ -e "${lidar_device}" || -L "${lidar_device}" ]]; then
    if ! lidar_real="$(realpath -e -- "${lidar_device}")"; then
      echo "ERROR: lidar device link is dangling: ${lidar_device}" >&2
      return 1
    fi
    if [[ "${lidar_real}" == "${left_real}" || "${lidar_real}" == "${right_real}" ]]; then
      echo "ERROR: lidar and Revo2 share a serial device; refusing hand control." >&2
      echo "  lidar=${lidar_device} -> ${lidar_real}" >&2
      echo "  left=${left_port} -> ${left_real}" >&2
      echo "  right=${right_port} -> ${right_real}" >&2
      return 1
    fi
  fi
  echo "SERIAL_OWNERSHIP_OK left=${left_real} right=${right_real} lidar=${lidar_real}"
}

feedback_failure() {
  local reason="$1"
  local inspection="" logs=""
  inspection="$(docker container inspect "${feedback_name}" \
    --format 'status={{.State.Status}} exit={{.State.ExitCode}} error={{.State.Error}}' \
    2>/dev/null || true)"
  logs="$(docker logs --tail 40 "${feedback_name}" 2>&1 || true)"
  # A failed hardware probe must not remain in an unless-stopped restart loop
  # hammering the serial buses. The bridge is read-only, so stopping it leaves
  # both hand command topics absent and is the fail-closed state.
  docker stop -t 1 "${feedback_name}" >/dev/null 2>&1 || true
  echo "ERROR: Revo2 feedback-only bridge is not ready: ${reason}" >&2
  [[ -z "${inspection}" ]] || echo "${inspection}" >&2
  [[ -z "${logs}" ]] || echo "${logs}" >&2
  return 1
}

feedback_container_running() {
  [[ "$(docker container inspect "${feedback_name}" \
    --format '{{.State.Running}}' 2>/dev/null || true)" == "true" ]]
}

verify_feedback_bridge() {
  if ! feedback_container_running; then
    feedback_failure "container exited immediately"
    return 1
  fi
  if [[ ! -r "${ros_setup}" ]]; then
    feedback_failure "ROS setup is missing: ${ros_setup}"
    return 1
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    feedback_failure "timeout command is unavailable"
    return 1
  fi

  # The external stop path may not have sourced ROS. Verify real messages,
  # not only a briefly running container or a stale discovery entry.
  set +u
  # shellcheck disable=SC1090
  source "${ros_setup}"
  set -u
  # The XR host installs FastDDS, while the bridge image uses CycloneDDS.
  # DDS interop is intentional. Clear any inherited container-only RMW choice
  # so the host proof uses its installed implementation on ROS domain 0.
  unset RMW_IMPLEMENTATION CYCLONEDDS_URI
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0
  local topic

  # First prove the bridge itself is publishing. Besides giving serial
  # initialization a bounded window, this avoids racing host DDS discovery
  # against a publisher that has not produced its first sample yet.
  for topic in /revo2/state/left /revo2/state/right; do
    if ! feedback_container_running; then
      feedback_failure "container exited while initializing ${topic}"
      return 1
    fi
    if ! docker exec "${feedback_name}" /bin/bash -lc \
        "source /opt/ros/jazzy/setup.bash && timeout --foreground 15s \
        ros2 topic echo '${topic}' std_msgs/msg/String --once \
        --qos-reliability best_effort --qos-durability volatile" \
        >/dev/null 2>&1; then
      feedback_failure "bridge did not publish a live message on ${topic}"
      return 1
    fi
  done

  # Then prove the host process used by the live controller can receive the
  # same samples across the Docker host-network boundary.
  for topic in /revo2/state/left /revo2/state/right; do
    if ! feedback_container_running; then
      feedback_failure "container exited while waiting for ${topic}"
      return 1
    fi
    if ! timeout --foreground 15s ros2 topic echo "${topic}" \
        std_msgs/msg/String --once \
        --qos-reliability best_effort --qos-durability volatile \
        >/dev/null 2>&1; then
      feedback_failure "XR host could not receive a live message from ${topic}"
      return 1
    fi
  done
  sleep 0.5
  if ! feedback_container_running; then
    feedback_failure "container exited during the stability check"
    return 1
  fi
}

case "${1:-}" in
  control)
    verify_serial_ownership
    docker stop "${feedback_name}" >/dev/null 2>&1 || true
    if docker container inspect "${control_name}" >/dev/null 2>&1; then
      docker start "${control_name}" >/dev/null
    else
      docker run -d \
        --name "${control_name}" \
        --restart unless-stopped \
        --network host \
        --privileged \
        --security-opt label=disable \
        -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -e CYCLONEDDS_URI=file:///opt/xr/config/cyclone_uri/ser7.cyclonedds.xml \
        -e PI05_REVO2_LEFT_PORT="${left_port}" \
        -e PI05_REVO2_RIGHT_PORT="${right_port}" \
        -v /opt/xr/config:/opt/xr/config \
        -v /home/xr/robocontrol_ws/revo2_vr_bridge:/bridge \
        -v /home/xr/.venvs/brainco-revo2:/opt/brainco-revo2 \
        -v /dev:/dev \
        "${image}" \
        /bin/bash -lc \
        'source /opt/ros/jazzy/setup.bash && export PYTHONPATH=/opt/brainco-revo2/lib/python3.12/site-packages:/bridge:${PYTHONPATH:-} && exec python3 /bridge/vr_trigger_revo2_bridge.py --hands both --left-port "${PI05_REVO2_LEFT_PORT}" --right-port "${PI05_REVO2_RIGHT_PORT}" --input-source absolute --command-rate 10 --duration-ms 100 --input-timeout 0.3 --min-absolute-delta 1'
    fi
    echo "Revo2 absolute bridge starting; controller will wait for both ROS subscribers."
    ;;

  feedback)
    verify_serial_ownership
    # Disable the command bridge first even when feedback recovery itself is
    # misconfigured. A missing read-only bridge must never leave absolute
    # command subscriptions alive.
    docker stop "${control_name}" >/dev/null 2>&1 || true
    if [[ ! -r "${feedback_script}" ]]; then
      echo "ERROR: feedback bridge source is missing: ${feedback_script}" >&2
      exit 1
    fi
    if docker container inspect "${feedback_name}" >/dev/null 2>&1; then
      docker start "${feedback_name}" >/dev/null
    else
      docker run -d \
        --name "${feedback_name}" \
        --restart unless-stopped \
        --network host \
        --privileged \
        --security-opt label=disable \
        -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -e CYCLONEDDS_URI=file:///opt/xr/config/cyclone_uri/ser7.cyclonedds.xml \
        -e PI05_REVO2_LEFT_PORT="${left_port}" \
        -e PI05_REVO2_RIGHT_PORT="${right_port}" \
        -v /opt/xr/config:/opt/xr/config \
        -v /home/xr/robocontrol_ws/revo2_vr_bridge:/bridge \
        -v /home/xr/.venvs/brainco-revo2:/opt/brainco-revo2 \
        -v /dev:/dev \
        -v "${root}:/app" \
        "${image}" \
        /bin/bash -lc \
        'source /opt/ros/jazzy/setup.bash && export PYTHONPATH=/opt/brainco-revo2/lib/python3.12/site-packages:/bridge:${PYTHONPATH:-} && exec python3 /app/revo2_feedback_only.py --left-port "${PI05_REVO2_LEFT_PORT}" --right-port "${PI05_REVO2_RIGHT_PORT}"'
    fi
    verify_feedback_bridge
    echo "Revo2 feedback-only bridge restored."
    ;;

  *)
    usage
    exit 2
    ;;
esac
