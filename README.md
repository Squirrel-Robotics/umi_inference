# UMI 推理服务端（5090）

此分支保存 5090 端的策略服务。它负责校验并恢复指定的 OpenPI checkpoint，
通过 ADB 启动 E6 头部相机视频流，以 WebSocket 提供策略推理服务，并将每次
真机请求记录到仓库目录之外。

## 启动

相机预处理模式不会保存在 Orbax checkpoint 中，因此启动时必须明确指定。
`full` 表示头部图像不裁剪，`roi` 表示采用训练时的 ROI 裁剪流程。

```bash
UMI_TASK_PROMPT='Put the object on the box ,then return it back.' \
bash /home/dzq/pi05_inference/start_umi_server.sh \
  /mnt/dzq/checkpoint/umi_task_v1/6000 \
  full
```

当前启动脚本的默认提示词是
`Put the object on the box ,then return it back.`。建议始终通过
`UMI_TASK_PROMPT` 传入与所选任务和 checkpoint 完全一致的训练提示词，
不要依赖默认值。

通用部署流程不使用 `deployment_profile.json`、checkpoint manifest，
也不使用 checkpoint 或哈希白名单。启动时仍会校验 checkpoint 目录结构、
30 维归一化统计、相机模式、有限值 warm-up 输出以及 `(H, 30)` action 合同。

## 调用链

```text
start_umi_server.sh
  -> start_umi_v2_server.sh
       -> umi_live_contract.py       # 通用 checkpoint/相机合同校验
       -> 启动 E6 Activity 并建立 ADB 端口转发
       -> serve_umi_v2_policy.py
            -> e6_tcp_cam_h.py       # E6 cam_h 帧历史与时间对齐
            -> camera_sync.py
            -> server_camera_sync.py
            -> camera_transport.py
            -> cam_high_roi.py       # full/ROI 图像预处理
            -> umi_policy_common.py  # OpenPI observation 变换
            -> inference_recorder.py
```

WebSocket 策略服务默认监听 `0.0.0.0:8000`。如果没有通过环境变量覆盖，
XR 客户端连接 `192.168.110.199:8000`。切换 checkpoint 前，先在 5090
前台终端按 `Ctrl+C` 停止当前服务。

## 主机依赖与生成数据

本仓库不包含 `/home/dzq/openpi` 及其虚拟环境、
`/home/dzq/openpi-assets`、`/mnt/dzq/checkpoint` 下的 checkpoint、
E6 Android 应用和 ADB 配置。真机推理请求默认记录到
`/home/dzq/pi05_inference_records`，该目录不会纳入 Git 版本管理。
