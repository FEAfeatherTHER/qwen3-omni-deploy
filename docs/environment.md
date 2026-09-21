# 环境与模型迁移

适用范围：Linux x86_64、Python 3.12.14、NVIDIA GPU。已验证的核心版本为 vLLM 0.16.0、PyTorch 2.9.1+cu128、transformers 4.57.6、qwen-omni-utils 0.0.9。服务器需要可用的 NVIDIA 驱动；Python 环境内安装 CUDA 12.8 运行库，无需修改系统 CUDA。

## 新建环境

已有 conda 时，在仓库目录执行：

```bash
# 在 tmux 内安装，断开 SSH 不影响下载。
tmux new -s qwen-env
bash scripts/create_env.sh --prefix "$PWD/.runtime/conda-env"
# 完成后按 Ctrl-b，再按 d 退出 tmux。
```

脚本仅创建指定的新目录，拒绝使用已有目录，不改动现有环境、shell 配置、全局 conda 或 pip 配置。安装中断后，不要把半成品环境当作已验证环境；保留日志，并指定另一个新目录重新执行。pip 下载缓存会继续复用。

`requirements.txt` 是直接依赖；`requirements.lock` 固定它们在原始已验证环境中的实际依赖闭包，共 188 个包，而非复制整个环境。`environment.yml` 只描述 conda 的 Python 与 pip 基础；完整安装应使用上述脚本。脚本正常解析依赖并执行 `pip check`，不会用 `--no-deps` 隐藏冲突。CUDA 专用包来自官方 PyTorch 源，其余包优先使用清华镜像；仅安装预编译包，避免新机器悄悄本地编译出不同产物。

依赖下载可能超过 10 GB，因此脚本在自身进程内清除代理，使用服务器网络。清华源需要能直接访问；脚本不会修改代理的系统设置。已存在的 pip/conda 下载缓存可复用，不复制旧环境的 `site-packages`。源环境残留的 setuptools 83.0.0 元数据不代表实际版本；锁文件采用其有效安装版本 80.10.2，满足 vLLM 的版本要求。另显式补齐 qwen-omni-utils 导入但未声明的 audioread 3.1.0，避免只通过依赖检查却无法导入。

若服务器没有 conda，可安装在新的用户目录：

```bash
# 以下路径必须尚不存在；无需 sudo，也无需运行 conda init。
mkdir -p .runtime
curl --noproxy '*' -fL \
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o .runtime/miniconda-installer.sh
bash .runtime/miniconda-installer.sh -b -p "$PWD/.runtime/miniconda"
CONDA_EXE="$PWD/.runtime/miniconda/bin/conda" \
  bash scripts/create_env.sh --prefix "$PWD/.runtime/conda-env"
```

请先确认安装脚本保存路径没有个人文件。[清华镜像使用说明](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)列出了官方镜像路径。

安装脚本最后执行依赖一致性检查和核心 Python 导入，不占用 GPU。实际四卡加载、HTTP 推理与音频验收另见 README；导入通过不代表 GPU 推理已经通过。

## 模型文件

模型目录不进入 Git。迁移时可复制已有完整模型，或者从 Hugging Face 下载。实际已验证目录包含 15 个权重分片，权重文件合计约 70.52 GB，因此应为模型与环境预留额外磁盘空间。

```bash
python3 scripts/check_model.py /path/to/Qwen3-Omni-30B-A3B-Instruct
```

该检查只依赖 Python 标准库，读取配置、词表、索引和权重文件头，核对全部张量名称、数据范围与文件尺寸，不加载权重，也不计算 SHA256。它能发现漏文件、截断、索引与文件头不匹配；不能检测不改变文件尺寸和结构的权重内容损坏。首次迁移后仍需实际推理验收。

需要下载时，在 tmux 中用新环境的 Python 运行：

```bash
tmux new -s qwen-model
.runtime/conda-env/bin/python scripts/download_model.py \
  /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint https://hf-mirror.com
```

默认仓库为 `Qwen/Qwen3-Omni-30B-A3B-Instruct`，默认站点为 `https://huggingface.co`。可用 `--revision COMMIT` 固定模型版本；脚本会将一次下载固定到同一个远端提交，结束时打印提交号。此参数控制新下载，已有完整模型直接通过本地检查并复用，不连接网络，因此不会声称已核对其来源或远端版本。

下载进程使用服务器网络，默认关闭代理和 Xet，使用 Hugging Face 的 HTTP 断点续传。中断后原样重跑；保留目标目录内 `.cache/huggingface` 的下载状态。缓存目录行为见 [Hugging Face 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。已有完整模型不需要联网；不完整目录内，尺寸相同的现有文件会复用，尺寸不符的现有文件会保留并报错，不自动覆盖。若本地文件损坏或混有其他版本，请检查后使用新的目标目录下载。

下载完成后自动执行同一套本地结构检查。大文件下载无需持续监控，可稍后查看 tmux 会话。

## 本次环境验证

2026-09-21 在 `.runtime/rebuild-env` 从空目录实际安装，最终 188 个锁定版本一致，`pip check` 及核心模型类导入通过；导入均来自新目录，未初始化 CUDA。结果记录于 `.runtime/environment-verification.json`。初次安装与未声明依赖的修正分别记录于 `.runtime/environment-rebuild-verified.log` 和 `.runtime/environment-rebuild-correction.log`；未复制旧环境的包。

真实模型 15 个分片的结构检查通过。下载续传使用真实 Hugging Face HTTP 客户端和本地 12MB 测试文件验证：模拟 10MB 处断线后续传，最终内容完全一致；没有重复下载已有 70GB 权重。
