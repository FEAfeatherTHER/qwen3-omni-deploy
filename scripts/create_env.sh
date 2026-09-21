#!/usr/bin/env bash
# Linux x86_64; no shell initialization or global conda/pip configuration changes.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_name=qwen3-omni
prefix=
selector=
while (($#)); do
  case "$1" in
    --name|--prefix)
      [[ -z "$selector" ]] || { echo 'Use either --name or --prefix, once.' >&2; exit 2; }
      selector=$1
      if [[ "$1" == --name ]]; then env_name="${2:?--name requires a name}"
      else prefix="${2:?--prefix requires a path}"; fi
      shift 2 ;;
    -h|--help) echo "Usage: bash scripts/create_env.sh [--name NAME | --prefix NEW_DIRECTORY]"
      echo 'Default: --name qwen3-omni (conda manages the installation directory).'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$prefix" && ( -e "$prefix" || -L "$prefix" ) ]]; then
  echo "Refusing destination that already exists: $prefix" >&2
  exit 2
fi
if [[ -z "$prefix" && ( ! "$env_name" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ || "$env_name" == base ) ]]; then
  echo 'Invalid environment name; base is reserved.' >&2; exit 2
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
if [[ -n "$prefix" ]]; then
  env_args=(--prefix "$prefix")
else
  conda_base="$("$conda_bin" info --base)"
  conda_info="$("$conda_bin" info --json)"
  existing="$("$conda_base/bin/python" -c '
import json,sys
from pathlib import Path
info=json.load(sys.stdin)
name=sys.argv[1]
candidates=[Path(p) for p in info["envs"] if Path(p).name == name]
candidates += [Path(p)/name for p in info["envs_dirs"]]
print(next((str(p) for p in candidates if p.exists() or p.is_symlink()), ""))
' "$env_name" <<< "$conda_info")"
  if [[ -n "$existing" ]]; then
    echo "Environment already exists: $existing. Reuse it or select a different --name; nothing changed." >&2
    exit 2
  fi
  env_args=(--name "$env_name")
fi
"$conda_bin" create -y "${env_args[@]}" --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  python=3.12.14 pip=26.2.1
run_python=("$conda_bin" run --no-capture-output "${env_args[@]}" python)
"${run_python[@]}" -m pip --isolated install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --only-binary=:all: -r "$repo_dir/requirements.lock"
"${run_python[@]}" -m pip check
"${run_python[@]}" - <<'PY'
import sys
import importlib.metadata as m
import torch, torchaudio, torchvision, transformers, vllm, qwen_omni_utils
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
for name in ('vllm', 'torch', 'torchaudio', 'torchvision', 'transformers', 'qwen-omni-utils'):
    print(f'{name}=={m.version(name)}')
print('CPU-only import smoke passed; no CUDA context created.')
print(f'Environment ready: {sys.prefix}')
PY
printf 'Activate with: conda activate %q\n' "${prefix:-$env_name}"
