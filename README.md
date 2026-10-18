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
