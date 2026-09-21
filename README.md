# Qwen3-Omni 本地部署

使用四张 NVIDIA GPU 部署 `Qwen/Qwen3-Omni-30B-A3B-Instruct`，提供文字与音频理解的 HTTP 服务。仅运行 Thinker、返回文字，跳过 Talker 和语音生成权重。客户端可通过 SSH 通道从另一台服务器调用，无需共享音频目录。

服务采用 [Qwen 官方的 vLLM 部署方式](https://github.com/QwenLM/Qwen3-Omni#vllm-usage)，模型来自 [Hugging Face 官方库](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)。仓库只保存部署代码和文档，模型与 conda 环境单独管理。

## 本机使用

本机的 `config.local.json` 已指向已有 `qwen3-omni` 环境及完整权重目录，使用 GPU `0,1,3,4`。该文件不会进入 Git。通用默认值仍为 GPU `0,1,2,3`。

```bash
cd /mnt/workspace/jingchong/code/qwen3-omni-deploy
python3 deploy.py status
python3 client.py --prompt '请用一句话介绍你自己。'
python3 client.py --prompt '请转写这段语音。' --audio /path/to/recording.wav
```

服务地址：`http://127.0.0.1:8001/v1`；模型名称：`Qwen3Omni-Instruct`。客户端只需 Python 3.10 或更新版本，不需激活模型环境。

## 新服务器部署

准备 Linux x86_64、conda、tmux 和可用的 NVIDIA 驱动。四卡验证硬件为 RTX 4090 D（每卡 24 GB）。其他型号需重新运行检查和推理验收；仅 GPU 数量相同不能保证可运行。

1. 将本仓库复制或克隆到新服务器。
2. 创建新的 conda 环境：

   ```bash
   tmux new -s qwen-env
   bash scripts/create_env.sh --prefix "$PWD/.runtime/conda-env"
   ```

   按 `Ctrl-b`，再按 `d` 可离开 tmux。安装使用清华镜像及对应 CUDA 的官方 PyTorch 包，不修改原环境或系统驱动。详细步骤及无 conda 时的安装方法见 [环境说明](docs/environment.md)。

3. 将完整权重复制到新路径，或在 tmux 中下载：

   ```bash
   .runtime/conda-env/bin/python scripts/download_model.py \
     "$PWD/models/Qwen3-Omni-30B-A3B-Instruct" --endpoint https://hf-mirror.com
   python3 scripts/check_model.py models/Qwen3-Omni-30B-A3B-Instruct
   ```

   权重约 70.52 GB。大文件下载使用服务器网络；下载中断后可重跑，不必持续监控。

4. 在仓库根目录新建 `config.local.json`：

   ```json
   {
     "env_prefix": ".runtime/conda-env",
     "model_path": "models/Qwen3-Omni-30B-A3B-Instruct",
     "gpus": "0,1,2,3"
   }
   ```

5. 确认四张卡空闲，检查并启动：

   ```bash
   python3 deploy.py doctor
   python3 deploy.py start
   python3 client.py --prompt '请只回答：服务正常。'
   ```

`start` 等待服务加载完成后才报告就绪。服务运行在独立 tmux 会话中，断开 SSH 后继续运行；服务器重启后需重新执行 `start`。

## 服务管理

```bash
python3 deploy.py config           # 查看有效配置
python3 deploy.py doctor           # 启动前检查；需要空闲端口和 GPU
python3 deploy.py start            # tmux 启动，默认最多等待 900 秒
python3 deploy.py status           # 查看进程与 HTTP 状态；未就绪时返回非零
python3 deploy.py logs             # 查看最近日志
python3 deploy.py logs --follow    # 持续查看日志
python3 deploy.py stop             # 停止此仓库启动的服务
python3 deploy.py run              # 前台运行，Ctrl+C 停止
```

日志和运行状态保存在 `.runtime/`。`doctor` 用于启动前检查，运行中请用 `status`；端口已占用或 GPU 显存不足时不会抢占其他服务。`stop` 检查记录的进程身份，不会按名称批量停止 Python 或 vLLM。

默认设置：四卡分担模型、BF16、32,768 上下文长度、最多 4 个并行请求、每次最多 4 段音频、显存使用比例 0.90。关闭图编译以简化启动；单段约 60 秒属于验收范围，不代表模型硬性上限，也不保证四个长音频请求同时达到同样性能。

全部配置项见 [config.example.json](config.example.json)。优先级为：内置默认值 → 本地配置文件 → 环境变量。相对模型、环境和运行目录始终按仓库根目录解释，不依赖终端当前目录。

```bash
QWEN_GPUS=0,1,2,3 QWEN_PORT=8002 python3 deploy.py start
python3 deploy.py --config /path/to/another.json config
```

常用覆盖项为 `QWEN_ENV_PREFIX`、`QWEN_MODEL`、`QWEN_GPUS`、`QWEN_PORT`、`QWEN_MAX_MODEL_LEN`、`QWEN_MAX_NUM_SEQS`、`QWEN_GPU_MEMORY_UTILIZATION`、`QWEN_STARTUP_TIMEOUT`。GPU 数量变更时同时调整 `tensor_parallel_size` 或 `QWEN_TP`；本仓库的实测配置是四卡。

一个运行目录只管理一个服务。多实例需分别设置 `runtime_dir`、`tmux_session`、端口和互不重叠的 GPU。

### 版本兼容说明

固定版本为 Python 3.12.14、vLLM 0.16.0、PyTorch 2.9.1+cu128、Transformers 4.57.6、qwen-omni-utils 0.0.9，沿用本机已运行组合，避免独立升级造成依赖变化。官方 Transformers 最新用法与本仓库固定的 vLLM 环境可能不同，不应混用安装命令。

当前 vLLM 完全关闭图片组件时会发生启动错误，因此内部保留 `image=1, video=0` 的兼容配置。[相关问题](https://github.com/vllm-project/vllm/issues/49384)。本仓库的客户端和验收仅涵盖文字、音频；这不意味着底层 HTTP 接口会拒绝所有图片请求。此设置不会启用 Talker。权重文件保留官方完整格式，由 vLLM 在加载时跳过语音生成部分。

## 从另一台服务器调用

在调用端复制 `client.py` 和 `scripts/tunnel.sh`，然后建立 SSH 通道：

```bash
bash scripts/tunnel.sh user@gpu-server
```

该终端保持运行；也可放入调用端的 tmux。另开终端：

```bash
curl --noproxy '*' -f http://127.0.0.1:18001/health
python3 client.py --base-url http://127.0.0.1:18001/v1 \
  --prompt '请转写这段语音。' --audio /client/path/recording.wav
```

音频字节通过 HTTP 发送，并经 SSH 通道传输；GPU 服务器不需要拥有 `/client/path/recording.wav`。服务与通道均只监听 `127.0.0.1`，无需对外开放模型端口。

等价 SSH 命令：

```bash
ssh -N -L 127.0.0.1:18001:127.0.0.1:8001 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 user@gpu-server
```

Python 调用方式：

```python
from client import chat, extract_text

response = chat(
    '比较这两段音频。', ['first.wav', 'second.wav'],
    base_url='http://127.0.0.1:18001/v1', timeout=300,
)
print(extract_text(response))
```

客户端默认等待完整文字结果，不自动重试。`--json` 可查看完整响应；`finish_reason=length` 表示达到输出长度限制，可调整 `--max-tokens`。更多参数见 [客户端说明](docs/client.md)。

纯文字 curl 示例：

```bash
curl --noproxy '*' -f http://127.0.0.1:18001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3Omni-Instruct","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":256,"stream":false}'
```

## 迁移与验收

代码和权重分别复制到新的空目录，避免混入旧机器配置。以下命令中的目标地址与路径需替换为实际值；不使用 `--delete`：

```bash
rsync -a --exclude='.runtime' --exclude='config.local.json' --exclude='models' \
  --exclude='__pycache__' ./ user@new-server:/new/path/qwen3-omni-deploy/
rsync -a --partial --info=progress2 --exclude='.parts' --exclude='.cache' \
  /source/path/Qwen3-Omni-30B-A3B-Instruct/ \
  user@new-server:/new/model/path/Qwen3-Omni-30B-A3B-Instruct/
```

模型复制建议在 tmux 内执行；勿复制来源目录中未完成下载的 `.parts` 临时数据。新服务器按上文重建 conda 环境及本地配置。

```bash
python3 -m unittest discover -s tests -v
python3 scripts/smoke_test.py --audio speech.wav --second-audio sound.wav \
  --long-audio recording-60s.wav --report .runtime/smoke-new.json
```

推理验收包括文字问答、真实语音转写、声音描述、四音频请求、约 60 秒音频和四个并行音频请求。报告保留模型回答与耗时，检查非空文字和完整结束；这属于服务可用性检查，不是识别准确率评测。验收记录见 [验证结果](docs/verification.md)。

若接替失败，停止新仓库拥有的服务，并使用此前保留的启动命令恢复旧服务；原环境、权重和原脚本均不修改。另一台调用服务器尚未指定，因此真实跨服务器 SSH 结果需在目标机器上用上述健康检查和音频命令验证。
