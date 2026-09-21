# 部署验证记录

## 已验证项目

验证日期：2026-09-21。机器：Linux x86_64，glibc 2.35，NVIDIA 驱动 570.133.07，GPU 为 RTX 4090 D（每卡 24 GB）。本机使用 GPU 0、1、3、4；通用默认配置为 0、1、2、3。

- 36 项自动测试通过，覆盖配置优先级、迁移路径、端口冲突及重启、进程身份、工作进程清理、磁盘写入失败后的清理、模型完整性、客户端字节保真和 SSH 命令参数。
- Python 编译检查与 Shell 语法检查通过。
- 15 个权重分片、28,010 个张量的索引、文件头及尺寸检查通过；分片总尺寸 70,523,299,202 字节。该检查不是内容哈希校验。
- 原环境中的实际四卡启动通过，服务只监听 `127.0.0.1:8001`，模型名称为 `Qwen3Omni-Instruct`。加载器明确跳过 `talker.` 与 `code2wav.` 权重。
- tmux 启动调用结束后，服务仍正常响应；重复 `start` 被拒绝且原进程身份未改变。
- 仓库复制到含空格的新目录后，从 `/tmp` 调用仍正确按新仓库目录解析相对路径。

## 实际推理

原环境的首次验收结果：

| 输入 | 结果 |
| --- | --- |
| 纯文字 | 返回“服务正常。” |
| 官方中文短语音 | 返回“甚至出现交易几乎停滞的情况。” |
| 官方咳嗽音频 | 正确描述为连续咳嗽 |
| 四段音频 | 按顺序区分语音与咳嗽 |
| 60 秒音频 | 正确说明语音内容重复出现 |
| 四个并行音频请求 | 全部返回非空、完整文字 |

短语音来自 [Qwen 官方语音识别示例](https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/speech_recognition.ipynb)，咳嗽来自 [官方快速入门](https://github.com/QwenLM/Qwen3-Omni#quickstart)。样本下载到本机 `.runtime/audio/`。60 秒样本由该短语音重复并截取为 60 秒，用于验证音频长度承载；不代表真实长录音准确率。验收未设性能承诺，单次耗时仅保存在本机报告。

## 环境重建与生命周期

独立 conda 环境从空目录通过 conda create 和 pip install 重建，未克隆旧环境或 site-packages。188 项锁定依赖版本全部匹配，pip check 无冲突；PyTorch、vLLM、Transformers、Qwen 模型类及 qwen-omni-utils 的导入来自新环境。

重建中修正了两个原环境掩盖的问题：源环境残留两份 setuptools 版本记录，锁定版本已校正为实际使用的 80.10.2；qwen-omni-utils 未声明其直接导入的 audioread，已显式增加 audioread 3.1.0。原环境不修改，其余核心版本保持批准方案中的值。首次安装错误日志保留，最终补齐安装和验证结果另存，避免将失败日志视为成功证据。

新环境在 `.runtime/relocated repo`（路径含空格）的仓库副本中完成四卡启动，全部九个真实文字/音频请求通过，包括 60 秒样本及四请求并行。

原环境及新环境服务均已验证停止，每次停止后 GPU 0、1、3、4 的空闲显存均为 24,089 MiB。根目录服务随后使用原 `qwen3-omni` 环境重新启动并保持常驻，最终音频转写正常返回。

后续状态：2026-09-21 23:48 按用户要求停止本机服务，确认进程已退出，四张卡各恢复 24,089 MiB 空闲显存。正式用途为目标服务器部署、本机通过 HTTP 调用；上述常驻状态仅记录首次交付时的验证结果。

独立代码审查发现并修复了两类异常清理问题：主进程先退出导致工作进程残留，以及状态文件写入失败导致模型进程遗留。相应测试均先复现失败，再验证修复。实际重启另外暴露了 TIME_WAIT（已关闭连接的暂存状态）导致端口检查误判的问题，已按 HTTP 服务的端口复用方式修正，并增加测试确认仍会拒绝真正被监听的端口。

## 本机证据文件

以下文件不进入 Git，迁移仓库不包含原机器的运行状态：

- `.runtime/unit-tests.log`：自动测试输出。
- `.runtime/gpu-smoke-initial.json`：原环境九个真实请求的回答、用量、结束原因及耗时。
- `.runtime/gpu-smoke-rebuilt.json`：独立重建环境的同组九个真实请求。
- `.runtime/lifecycle-acceptance.json`：停止、显存释放、迁移启动和最终恢复结果。
- `.runtime/final-status.json`、`.runtime/final-audio-response.json`：最终常驻服务状态及音频响应。
- `.runtime/lifecycle-precheck.json`：重复启动保护与独立运行结果。
- `.runtime/relocation-check.json`：迁移目录路径检查。
- `.runtime/environment-rebuild-verified.log`：独立 conda 环境安装日志。
- `.runtime/environment-rebuild-correction.log`：补齐 audioread 后的最终安装及导入检查。
- `.runtime/environment-verification.json`：188 个锁定版本及新环境导入来源的最终核对。
- `.runtime/server-*.log`：实际服务启动与运行日志。

## 跨服务器 SSH

通道脚本已用替代 SSH 命令验证本机绑定、目标端口、保活和失败退出参数。另一台服务器尚未指定，未宣称跨服务器连接已实测。目标服务器可用后，在调用端建立通道，并按 README 执行健康检查与音频请求即可验收。
