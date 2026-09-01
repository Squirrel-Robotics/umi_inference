# UMI 机器人客户端（XR）

此分支保存 XR 端真机客户端。XR 负责采集并对齐双腕相机图像与机器人状态，
向 5090 请求策略推理，解码每个 Delta EEF action chunk，并向机器人下发
双臂绝对位姿目标和灵巧手绝对目标。

Checkpoint 选择、头部相机预处理模式和任务提示词均由 5090 服务合同决定；
XR 启动脚本只接受一个可选的动作播放频率参数。

## 启动与停止

先在 5090 上启动选定的 checkpoint。确认策略服务已经就绪后，在 XR 上运行：

```bash
bash /home/xr/pi05_umi_inference/start_umi_robot.sh 10
```

正常停止时，直接在该前台终端按 `Ctrl+C`。如果启动终端已经不可用，则运行：

```bash
bash /home/xr/pi05_umi_inference/stop_umi_robot.sh
```

启动流程会先完成不下发命令的策略检查，然后向左右灵巧手各发送一次归零目标，
确认 ACK 与反馈后才获取机械臂控制权。启动前必须清空机器人周围空间。
启动脚本不会自动运行 `/home/xr/pi05/zero_arms_z086.py`，机械臂回零仍需操作者
提前手动完成。

## 当前调度默认值

- 模型 action 播放频率：默认 `10 Hz`，可由启动脚本参数覆盖。
- 计划切换 chunk 的位置：`PI05_CHUNK_SIZE=40`。
- 异步推理提前量：当前提交的启动脚本默认
  `PI05_INFERENCE_LEAD_STEPS=10`。
- 机械臂插值命令频率：`PI05_ARM_COMMAND_RATE_HZ=50`。

机械臂命令频率只控制 SE(3) 中间插值命令，不会改变模型 action 或 chunk 的
播放速度。将它设置为与 action 播放频率相同，即可关闭中间插值点。

根据当前 10 Hz/H50 真机链路的实测结果，可使用下面的低延迟覆盖参数：

```bash
PI05_CHUNK_SIZE=40 \
PI05_INFERENCE_LEAD_STEPS=5 \
PI05_ARM_COMMAND_RATE_HZ=30 \
bash /home/xr/pi05_umi_inference/start_umi_robot.sh 10
```

以上参数只作为运行时覆盖示例写入文档，没有静默修改源码默认值。

## 调用链

```text
start_umi_robot.sh
  -> runtime/policy_profile_preflight.py   # 策略合同预检
  -> runtime/wrist_camera_bridge.sh        # 双腕相机
  -> runtime/revo2_bridge.sh control       # 灵巧手控制桥
  -> runtime/umi_ros_live_chunked.py
       -> robot_observation.py             # 图像与机器人状态采集
       -> camera_sync.py + camera_transport.py
       -> low_latency_policy_client.py     # 请求 5090 策略
       -> async_chunk_scheduler.py         # active/standby 异步 chunk 调度
       -> se3_actions.py                   # Delta EEF 解码
       -> robot_command_interpolation.py   # SE(3) 命令插值
       -> inference_stop.py                # Ctrl+C 与停止流程

停止或异常退出
  -> 移除真机命令使能门
  -> 保持当前反馈；必要时调用独立急停辅助程序
  -> runtime/revo2_bridge.sh feedback
```

当前启动脚本还会应用 `start_umi_robot.sh --help` 中说明的现场专用左臂执行锚点
偏移。每次真机运行前都应先阅读该帮助信息。

ROS Jazzy、XR Python 环境、X2 SDK、Revo2 bridge 依赖、Docker 镜像和硬件
设备配置均属于 XR 主机环境，不包含在此分支中。日志、锁文件、停止代次文件、
备份和 staging 目录均已排除，不会提交到 Git。
