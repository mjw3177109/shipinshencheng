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
