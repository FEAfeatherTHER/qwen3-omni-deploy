#!/usr/bin/env bash
# Linux x86_64; no shell initialization or global conda/pip configuration changes.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="$repo_dir/.runtime/conda-env"
while (($#)); do
  case "$1" in
    --prefix) prefix="${2:?--prefix requires a path}"; shift 2 ;;
    -h|--help) echo "Usage: bash scripts/create_env.sh [--prefix NEW_DIRECTORY]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -e "$prefix" || -L "$prefix" ]]; then
  echo "Refusing destination that already exists: $prefix" >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This lock supports Linux x86_64 only.' >&2; exit 2
fi
# The CUDA dependency set can exceed 10 GB: use the server network by default.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
export PYTHONNOUSERSITE=1
unset PYTHONPATH
conda_bin="${CONDA_EXE:-}"
if [[ -z "$conda_bin" ]]; then conda_bin="$(command -v conda || true)"; fi
if [[ -z "$conda_bin" ]]; then
  echo 'conda not found. Install Miniconda as described in docs/environment.md.' >&2
  exit 2
fi
"$conda_bin" create -y --prefix "$prefix" --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  python=3.12.14 pip=26.2.1
"$prefix/bin/python" -m pip --isolated install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --only-binary=:all: -r "$repo_dir/requirements.lock"
"$prefix/bin/python" -m pip check
"$prefix/bin/python" - <<'PY'
import importlib.metadata as m
import torch, torchaudio, torchvision, transformers, vllm, qwen_omni_utils
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
for name in ('vllm', 'torch', 'torchaudio', 'torchvision', 'transformers', 'qwen-omni-utils'):
    print(f'{name}=={m.version(name)}')
print('CPU-only import smoke passed; no CUDA context created.')
PY
printf 'Environment ready: %s\n' "$prefix"
