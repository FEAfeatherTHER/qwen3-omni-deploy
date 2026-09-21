# 文本与音频客户端

`client.py` 只需要 Python 3.10 或更新版本的标准库。可单独复制到其他机器，无需安装模型或 vLLM。默认地址为 `http://127.0.0.1:8001/v1`，模型名为 `Qwen3Omni-Instruct`。

```bash
python3 client.py --prompt '请用一句话介绍你自己。'
python3 client.py --prompt '请转写这段音频。' --audio '/path/to/录音.wav'
python3 client.py --prompt '比较这两段音频。' --audio first.wav --audio second.mp3
python3 client.py --prompt '请简短回答。' --system '使用中文。' --max-tokens 256 --json
```

默认输出回答文本。`--json` 输出完整响应，包括用量与 `finish_reason`。若 `finish_reason` 是 `length`，回答达到输出上限，可按需调整 `--max-tokens`，默认值为 1024。

`--audio` 可以重复使用，单次最多 4 个文件。文件从运行客户端的机器读取，其原始字节通过 HTTP 发送，不要求服务端存在同名路径。扩展名决定媒体类型，客户端不转码；支持 `.wav`、`.mp3`、`.flac`、`.ogg`、`.opus`、`.m4a`、`.mp4`、`.aac`、`.aiff`、`.aif`、`.webm`，实际解码能力以服务端为准。当前验收范围为每段不超过约 60 秒，这不是模型的硬性时长限制。

默认等待超时为 300 秒，可用 `--timeout` 调整。请求使用非流式响应，不自动重试；网络错误、HTTP 错误、缺失音频文件或异常响应均以非零状态退出。客户端忽略环境中的 HTTP 代理配置，且不会修改环境变量。运行时收到完整响应后才输出内容。

## 从另一台机器访问

在客户端机器运行：

```bash
bash scripts/tunnel.sh user@model-server
```

该命令保持前台运行，也可放在 tmux 中。默认将客户端 `127.0.0.1:18001` 转发到模型服务器 `127.0.0.1:8001`；在另一个终端调用：

```bash
python3 client.py --base-url http://127.0.0.1:18001/v1 \
  --prompt '请转写这段音频。' --audio ./local-recording.wav
```

支持 SSH 配置中的主机别名与自定义端口：

```bash
bash scripts/tunnel.sh model-server --ssh-port 2222 --local-port 18002 --remote-port 8001
```

SSH 仍按本机现有密钥、agent 或交互式认证工作，脚本不存储凭据。按 Ctrl-C 关闭隧道。隧道脚本已用替代 SSH 命令验证参数；真实跨服务器连接需有可用远程机器后验证。

## 在 Python 中调用

```python
from client import ClientError, chat, extract_text

try:
    response = chat(
        '比较这两段音频。',
        ['first.wav', 'second.wav'],
        base_url='http://127.0.0.1:18001/v1',
        timeout=300,
        max_tokens=1024,
    )
    print(extract_text(response))
    print(response.get('usage'))
except ClientError as error:
    print(f'请求失败：{error}')
```

`chat(prompt, audio_paths=(), *, base_url=..., model=..., timeout=300, max_tokens=1024, system=None)` 返回完整响应字典。`extract_text(response)` 提取第一条回答文本。
