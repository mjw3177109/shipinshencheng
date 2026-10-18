# MiniMax H3 — Kaggle T4×2 视频生成

项目入口：[Kaggle 自动生成 Notebook](kaggle/MiniMax_H3_Kaggle_T4x2_AutoVideo.ipynb)。

使用方法：登录 Kaggle → New Notebook → File/Import Notebook 上传上面的 ipynb；开启 **GPU T4 x2** 和 **Internet**；从上到下执行。Notebook 会检查 GPU、克隆 ComfyUI、下载 H3 INT8 主模型 / INT8 文本编码器 / 视频和音频 VAE / 8-step Turbo LoRA、启动本机服务、通过 ComfyUI API 提交视频任务，并把 MP4 保存到 `/kaggle/working/h3_videos/`。

### 已知限制
- 尚未在真实 Kaggle T4×2 GPU 会话上验收，显存、主机内存、磁盘空间或模型版本可能导致失败。
- **单条视频使用 GPU 0；GPU 1 并未参与该视频的协同计算**。绝不把两张 16GB 的显存错误地当成 32GB 共享显存。
- 当前默认测试规格为 384×224、22 帧，低于模型常用训练时长，仅用于验证推理链路。按需调整提示词、尺寸及帧数，帧数规则为 `17k+5`。模型实际输出和可用性待验收。
- 预估至少 48 GB 磁盘空余才能开始下载，运行期间还需要留出余量。Kaggle 临时目录非持久。
- **GitHub 仓库存放代码和 Notebook，不适合直接上传视频和大模型**。最终 MP4 存在 Kaggle Output，需要用户自行下载或另外配置存储。

本仓库不会收集 Kaggle 密钥，不启用无密码公网隧道。仅在合规范围内使用模型。

## 从 GitHub 推送并运行 Kaggle

- 入口：[`Publish MiniMax H3 to Kaggle`](https://github.com/mjw3177109/shipinshencheng/actions/workflows/publish-kaggle.yml)。
- 在本仓库的 **Settings → Secrets and variables → Actions → Repository secrets** 配置名称**完全一致**的 Secret：`shipin`（值为 Kaggle API Token，而不是凭据名称）。
- 选择 **Run workflow**。个人 Token 可以在输入框 `kaggle_username` 填 Kaggle 用户名；若已能通过 Token introspection 解析用户名，也可以留空。
- `run_video = true` 会推送 Notebook 并请求 T4×2 实际运行；`false` 只上传，不执行 Notebook。仓库代码不包含 Token。
- 查看 Actions 作业日志以获得 Kaggle Notebook 链接。**Github 工作流提交成功与 Kaggle 视频渲染成功是两回事**；需在 Kaggle 的 Runs/Output 查看任务是否完成。

> Kaggle Notebook 中同名的 Secret 不会自动变成 GitHub Actions Secret。若 `shipin` 只配置在 Kaggle，请在本仓库也单独配置 Actions Secret；不要把密钥发到聊天、提交进源码或粘贴到公开 Issue。

## 2026-10-18：三个真实工作流接入与 Kaggle 验收

- 参考 API 工作流：[`workflows/source/t2v.json`](workflows/source/t2v.json)、[`i2v.json`](workflows/source/i2v.json)、[`r2v.json`](workflows/source/r2v.json)。
- Notebook 默认 **T2V**（用于第一次无需图片的短测试）；I2V/R2V 不会被偷换。对应参考图放入 Kaggle `/kaggle/working/h3_inputs/`，文件名与 JSON 保持一致。
- 启动脚本 `kaggle/bootstrap_h3.py` 按模式下载不同模型；`kaggle/run_from_sources.py` 读取原始工作流提示词并在 ComfyUI API 上编译/提交当前模式的低显存测试图。
- GPU 策略：申请 Kaggle T4 ×2，但 ComfyUI 单次推理使用 GPU0。**GPU1 尚未作为同一视频的分布式计算 GPU**。
- 下载体积按 Hugging Face 模型元数据检查；Kaggle 本地磁盘或主机 RAM 不足时可能报错。MP4、日志、`render_report.json` 置于 `/kaggle/working/h3_videos/`。
- [首次上传并启动的 GitHub Actions 记录](https://github.com/mjw3177109/shipinshencheng/actions/runs/39120091506)；[Kaggle 私有 Notebook](https://www.kaggle.com/code/mjw3177109/minimax-h3-t4x2-shipin)。上传成功不代表 MP4 已生成。
- [只读取 Kaggle 任务状态](https://github.com/mjw3177109/shipinshencheng/actions/workflows/check-kaggle.yml)：不会重新提交 GPU 计算任务。

## 小硬盘模式：Compact H3（2026-10-18 增量补丁）

**默认开启** `H3_STORAGE_PROFILE=compact`。不需要重新输入 Kaggle Token，Notebook 在 Kaggle 启动后会先从本仓库获取最新版脚本。

| 配置 | FL2VA/T2V/I2V 单模型全套大约大小 | 特点 |
| --- | --- | --- |
| `balanced` | 53.5 GB | 原配：INT8 DiT + INT8 Qwen3-VL 编码器 + INT8 视频 VAE |
| **`compact`（默认）** | **42.0 GB** | INT8 DiT + 15.69GB NVFP4-AWQ 编码器 + INT8 视频 VAE |
| `experimental-w6a8` | 37.0 GB | 改用 16GB W6A8 DiT；T4 兼容/速度未经验收 |

以上是**模型文件磁盘体积估算**，不是显存或主机 RAM 用量。R2V `balanced` 原来还使用 5.21GB 的 fp16 视频 VAE，因此其原体积更大；`compact` 统一用 2.81GB INT8 视频 VAE。R2V 仍需自己的 Ref2VA 主模型，永不偷换成 I2V。Turbo LoRA 保留；如设 `H3_TURBO=0` 可少下载约 1.96GB，但采样步数恢复到 20，预计更慢。

**Kaggle 限制：** 官方 Notebook 资源文档记录 T4×2 配置 **29GB CPU RAM**、`/kaggle/working` **20GB 自动保存空间**，另外有不持久化的 scratch 空间；所以即使精简到 37–42GB，也不代表一定能在免费 Kaggle 生成 H3。启动前 `h3_preflight.json` 将展示实际磁盘需求和可用空间，脚本会在下载前拒绝容量不足的任务。GPU0 单卡推理未变，GPU1 仍未参与同一片视频计算。

**其他更小选项：** 社区另有 GGUF Q2/Q3 模型（约 6–9 GiB 扩散模型）及 GGUF 文本编码器（Q2 ~12.2 GiB），但要求另装兼容的 ComfyUI-GGUF-H3 加载器和重做 loader；不同 GGUF fork 对 diffusion K-quants 支持不一致，目前**未启用**，不把它当成稳定视频生产模式。

参考来源：
- https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models
- https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders
- https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae
- https://www.kaggle.com/docs/notebooks
- https://github.com/kyuz0/ComfyUI-GGUF-H3

## Ultra-Q2 GGUF 超小版（默认，新增 2026-10-18）

最新 Notebook 默认 `BACKEND='ultra-q2'`，用于替代此前约 42GB 的 ComfyUI Compact 版。它使用 **stable-diffusion.cpp** 的 GGUF H3 推理后端（不是把 GGUF 直接交给原 UNETLoader），从原 JSON 读取提示词和图片连接。**T2V/I2V 只用 FL2VA、R2V 只用 Ref2VA，不换模式。**

| 部分 | 仓库 / 文件 | 预计权重大小 |
| --- | --- | ---: |
| FL2VA DiT | `unsloth/MiniMax-H3-GGUF / minimax_h3_fl2va_pruned-Q2_K.gguf` | 6.26 GiB |
| R2V DiT（仅 R2V 时下载，不能与 FL2VA 同时作为必须项） | `unsloth/MiniMax-H3-GGUF / minimax_h3_ref2va_pruned-Q2_K.gguf` | 6.22 GiB |
| Qwen3-VL TE | `realrebelai/MiniMax-H3_GGUFs / qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | 约 7.91 GiB |
| 视频 VAE | `Comfy-Org/MiniMax-H3 / vae/minimax_h3_video_vae_fp16.safetensors` | 约 4.85 GiB |
| 音频 VAE | `Comfy-Org/MiniMax-H3 / vae/minimax_h3_audio_vae_fp32.safetensors` | 约 0.56 GiB |
| **仅一套模式合计** | **无 Turbo LoRA，20 采样步** | **约 19.6 GiB（21GB）** |

- `kaggle/ultra_gguf_h3.py` 提供下载前远程文件元数据校验、磁盘预检、CUDA SM75 编译、两张 T4 的实验性分层计算和带音频的视频输出。
- 多卡（实验）：`--backend diffusion=cuda0&cuda1,te=cpu,vae=cuda0`；此为 **stable-diffusion.cpp 真正支持的双设备 layer split**，不是两张卡显存自动合并。H3/T4 实际速度与稳定性仍需 Kaggle 实测；若不兼容，Notebook 的 `SINGLE_GPU=True` 可回到单卡。
- 初次测试 `t2v`、384×224、22 帧、20 步。以后选 `i2v` 或 `r2v`，必须把原工作流指定的真实参考图片放入 `/kaggle/working/h3_inputs/`；不得用占位图片替代。
- 模型及源码放到临时 `/kaggle/temp/h3_gguf_ultra/`；视频、构建日志、`ultra_report.json` 放到 `/kaggle/working/h3_videos/`。需要 **模型约 21GB + 编译及解压临时空间约 7GiB**，下载前检查 scratch 文件系统，不能保证 Kaggle 有足够空间和 RAM。
- `BACKEND='comfy-compact'` 保留旧版 ComfyUI 运行方式，不会删原工作流。
- 静态 CI: [Ultra Q2 验证](https://github.com/mjw3177109/shipinshencheng/actions/workflows/validate-h3-ultra.yml)；该结果不能取代 Kaggle GPU 验收。
- 模型遵守 MiniMax H3 社区协议及相应量化来源许可证。Q2 会降低画质，尤其有文字细节时更明显。文档计算的数值为模型磁盘容量，非显存峰值。

模型与双卡后端文档：
- https://huggingface.co/unsloth/MiniMax-H3-GGUF
- https://huggingface.co/realrebelai/MiniMax-H3_GGUFs
- https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/minimax_h3.md
- https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/backend.md
