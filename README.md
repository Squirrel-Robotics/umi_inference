# UMI inference server (5090)

This branch contains the 5090-side policy server. It validates and restores a
selected OpenPI checkpoint, starts the E6 head-camera stream through ADB, serves
the policy over WebSocket, and records each live request outside the repository.

## Start

The camera preprocessing mode is required because it is not stored in the
Orbax checkpoint. Choose `full` for an uncropped head image or `roi` for the
training-time ROI pipeline.

```bash
UMI_TASK_PROMPT='Put the object on the box ,then return it back.' \
bash /home/dzq/pi05_inference/start_umi_server.sh \
  /mnt/dzq/checkpoint/umi_task_v1/6000 \
  full
```

The current launcher default prompt is
`Put the object on the box ,then return it back.`. Set `UMI_TASK_PROMPT` to the
exact prompt used for the selected task/checkpoint instead of relying on the
default.

The generic deployment path intentionally does not use
`deployment_profile.json`, checkpoint manifests, or checkpoint/hash allowlists.
It still validates checkpoint layout, 30-D normalization statistics, camera
mode, finite warm-up output, and the `(H, 30)` action contract.

## Call chain

```text
start_umi_server.sh
  -> start_umi_v2_server.sh
       -> umi_live_contract.py       # generic checkpoint/camera contract
       -> ADB E6 activity + port forward
       -> serve_umi_v2_policy.py
            -> e6_tcp_cam_h.py       # E6 cam_h history and alignment
            -> camera_sync.py
            -> server_camera_sync.py
            -> camera_transport.py
            -> cam_high_roi.py       # full/ROI preprocessing
            -> umi_policy_common.py  # OpenPI observation transform
            -> inference_recorder.py
```

The WebSocket policy listens on `0.0.0.0:8000` by default. The XR client uses
`192.168.110.199:8000` unless overridden. Stop the foreground server with
`Ctrl+C` before selecting another checkpoint.

## Host dependencies and generated data

The repository does not include `/home/dzq/openpi`, its virtual environment,
`/home/dzq/openpi-assets`, checkpoints under `/mnt/dzq/checkpoint`, the E6 Android
application, or ADB configuration. Live request records default to
`/home/dzq/pi05_inference_records` and are intentionally not versioned.
