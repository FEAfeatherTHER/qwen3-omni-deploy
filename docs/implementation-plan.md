# Qwen3-Omni Deployment Implementation Plan

> Execution: superpowers:executing-plans; the user approved the conversation plan and requested implementation on 2026-09-21.

**Goal:** Portable conda-based, four-GPU Thinker HTTP service and text/audio client.

**Architecture:** vLLM's existing OpenAI-compatible API, tmux lifecycle, standard-library deployment manager and HTTP client. Model files remain outside Git. Machine-specific paths belong in ignored config.local.json.

**Tech Stack:** Python 3.12.14, vLLM 0.16.0, torch 2.9.1+cu128, transformers 4.57.6, qwen-omni-utils 0.0.9, conda, tmux, SSH.

## Constraints and decisions

- The approved plan in the conversation is the specification. Work in the user's new, empty standalone repository; no worktree is needed.
- Do not alter the existing environment, weights, or active evaluation. Take over GPU 0,1,3,4 and port 8001 only after evaluation completes.
- Defaults: GPU 0,1,2,3; port 8001; loopback host; model alias Qwen3Omni-Instruct; BF16; TP=4; 32768 context; 4 sequences; 0.90 memory utilization; eager mode; audio=4,image=1,video=0 compatibility setting.
- Text/audio clients only; up to 60 seconds per audio is the tested scope, not an invented hard model limit. No Talker. No automatic startup after reboot.
- HTTP audio carries file bytes, never client filesystem paths. SSH forwarding is initiated on the client server.
- Test-only files and new-environment caches stay in .runtime. Record external-server SSH verification as pending until a remote server is supplied.

## Task 1: Environment and model portability

- [x] Create conda installation scripts, lean pinned requirements, full resolved lock, environment guidance; reconstruct a fresh environment without cloning existing site-packages.
- [x] Add model-index/config/shard checks and resumable Hugging Face download with endpoint control and server-network settings.
- [x] Verify pip check, versions, model imports, model file checks, and report artifacts.

## Task 2: Service lifecycle

- [x] Write failing tests for config precedence, relative paths after relocation, invalid GPU/port inputs, occupied-port checks and process ownership.
- [x] Implement config and deployment manager with doctor/run/start/status/stop/logs, loopback-only API, tmux session and per-run logs.
- [x] Avoid terminating unrelated processes or treating a different server's health as successful startup. Wait for this server's readiness and exit nonzero on startup failure.
- [x] Verify unit tests and real lifecycle once evaluation ends.

## Task 3: HTTP client and SSH

- [x] Write local HTTP-server tests for exact audio bytes, multiple audio inputs, non-ASCII text, HTTP failures, no retry, and invalid responses.
- [x] Implement standalone client.py and scripts/tunnel.sh without server-side shared paths or heavyweight client dependencies.
- [x] Expose base URL, model, prompt, repeated audio paths, timeout and JSON output; default timeout 300 seconds.

## Task 4: Documentation and GPU acceptance

- [x] Write README with local setup, migration, model copy/download, service controls, SSH and Python/curl examples, compatibility limits and rollback.
- [x] Confirm old evaluation completion, load Thinker on four GPUs, exercise text, real audio, multi-audio, ~60s audio and four concurrent requests.
- [x] Test tmux detach, stop and restart; leave service ready. Independently review code, fix material findings and rerun necessary checks.
- [x] Commit source and verification summary, excluding machine state and large artifacts.

## Review focus

Missing weight shards; port belongs to another service; stale PID/session metadata; paths with spaces after migration; a failed HTTP request must not silently repeat expensive inference. Owners exercise these cases in their tests or acceptance commands.
