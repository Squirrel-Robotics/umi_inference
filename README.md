# UMI robot client (XR)

This branch contains the XR-side live client. XR gathers synchronized wrist
images and robot state, requests the 5090 policy, decodes each Delta EEF chunk,
and publishes absolute dual-arm and dexterous-hand targets.

Checkpoint selection, head-camera preprocessing, and the task prompt belong to
the 5090 server contract; the XR launcher takes only an optional playback rate.

## Start and stop

Start the selected checkpoint on 5090 first. After its policy server is ready,
run on XR:

```bash
bash /home/xr/pi05_umi_inference/start_umi_robot.sh 10
```

Normal stop is `Ctrl+C` in that foreground terminal. If the terminal is no
longer available, run:

```bash
bash /home/xr/pi05_umi_inference/stop_umi_robot.sh
```

Startup performs no-command policy checks, zeros both dexterous hands once,
confirms ACK and feedback, then takes arm authority. Clear the workspace before
starting. The launcher does not run `/home/xr/pi05/zero_arms_z086.py`; arm zeroing
remains an explicit operator step.

## Current scheduling defaults

- Model/action playback rate: `10 Hz` unless the launcher argument overrides it.
- Planned chunk boundary: `PI05_CHUNK_SIZE=40`.
- Async inference lead: `PI05_INFERENCE_LEAD_STEPS=10` in the committed launcher.
- Interpolated arm command rate: `PI05_ARM_COMMAND_RATE_HZ=50`.

The arm-command rate only inserts SE(3) intermediate commands and does not
change model-action timing. Set it equal to the playback rate to disable those
intermediate commands.

For the measured 10 Hz/H50 deployment, a lower-latency operator override is:

```bash
PI05_CHUNK_SIZE=40 \
PI05_INFERENCE_LEAD_STEPS=5 \
PI05_ARM_COMMAND_RATE_HZ=30 \
bash /home/xr/pi05_umi_inference/start_umi_robot.sh 10
```

This override is documented but is not silently made the source default.

## Call chain

```text
start_umi_robot.sh
  -> runtime/policy_profile_preflight.py
  -> runtime/wrist_camera_bridge.sh
  -> runtime/revo2_bridge.sh control
  -> runtime/umi_ros_live_chunked.py
       -> robot_observation.py
       -> camera_sync.py + camera_transport.py
       -> low_latency_policy_client.py
       -> async_chunk_scheduler.py
       -> se3_actions.py
       -> robot_command_interpolation.py
       -> inference_stop.py

stop or error
  -> remove live command gate
  -> hold current feedback / emergency helper when required
  -> runtime/revo2_bridge.sh feedback
```

The committed launcher also applies the current site-specific left-arm execution
anchor offset documented by `start_umi_robot.sh`. Read the launcher's `--help`
before live use.

ROS Jazzy, the XR Python environment, the X2 SDK, Revo2 bridge dependencies,
Docker images, and hardware device configuration are host dependencies and are
not included in this branch. Logs, locks, stop-generation files, backups, and
staging directories are intentionally ignored.
