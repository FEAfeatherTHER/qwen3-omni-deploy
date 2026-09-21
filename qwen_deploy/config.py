"""Portable configuration; relative paths always belong to the repository."""
import json
import os
from pathlib import Path
import re

DEFAULTS = {
    'env_name': 'qwen3-omni', 'env_prefix': '',
    'model_path': 'models/Qwen3-Omni-30B-A3B-Instruct',
    'gpus': '0,1,2,3', 'tensor_parallel_size': 4,
    'host': '127.0.0.1', 'port': 8001, 'model_name': 'Qwen3Omni-Instruct',
    'max_model_len': 32768, 'max_num_seqs': 4, 'max_audio': 4,
    'gpu_memory_utilization': 0.90, 'runtime_dir': '.runtime',
    'tmux_session': 'qwen3-omni', 'startup_timeout': 900,
}
ENV_KEYS = {
    'QWEN_ENV_NAME': 'env_name', 'QWEN_ENV_PREFIX': 'env_prefix',
    'QWEN_MODEL': 'model_path', 'QWEN_GPUS': 'gpus', 'QWEN_PORT': 'port',
    'QWEN_TP': 'tensor_parallel_size', 'QWEN_MODEL_NAME': 'model_name',
    'QWEN_MAX_MODEL_LEN': 'max_model_len', 'QWEN_MAX_NUM_SEQS': 'max_num_seqs',
    'QWEN_MAX_AUDIO': 'max_audio', 'QWEN_GPU_MEMORY_UTILIZATION': 'gpu_memory_utilization',
    'QWEN_RUNTIME_DIR': 'runtime_dir', 'QWEN_TMUX_SESSION': 'tmux_session',
    'QWEN_STARTUP_TIMEOUT': 'startup_timeout',
}


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'未知配置项: {", ".join(sorted(unknown))}')
    for key, default in DEFAULTS.items():
        value = config[key]
        if type(value) is not type(default):
            # JSON 1 is a valid floating-point value but a bool is never a number.
            if isinstance(default, float) and type(value) is int:
                config[key] = float(value)
            else:
                raise ValueError(f'{key} 类型应为 {type(default).__name__}')
    gpus = config['gpus'].split(',')
    if not all(re.fullmatch(r'\d+', part) for part in gpus):
        raise ValueError('gpus 应为逗号分隔的非负 GPU 编号')
    ids = [int(part) for part in gpus]
    if len(set(ids)) != len(ids) or len(ids) != config['tensor_parallel_size']:
        raise ValueError('GPU 编号不得重复，数量必须等于 tensor_parallel_size')
    if config['host'] != '127.0.0.1':
        raise ValueError('服务仅绑定 127.0.0.1；远程调用请使用 SSH 通道')
    if not 1 <= config['port'] <= 65535:
        raise ValueError('port 必须在 1–65535 之间')
    for key in ('tensor_parallel_size', 'max_model_len', 'max_num_seqs', 'max_audio', 'startup_timeout'):
        if config[key] < 1:
            raise ValueError(f'{key} 必须为正整数')
    if not 0 < config['gpu_memory_utilization'] < 1:
        raise ValueError('gpu_memory_utilization 必须在 0 和 1 之间')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['tmux_session']):
        raise ValueError('tmux_session 仅支持字母、数字、下划线和连字符')
    for key in ('model_path', 'runtime_dir', 'model_name', 'env_name'):
        if not config[key].strip():
            raise ValueError(f'{key} 不能为空')


def load_config(root, config_path=None, environ=None):
    root = Path(root).resolve()
    environ = os.environ if environ is None else environ
    explicit = config_path or environ.get('QWEN_CONFIG')
    path = Path(explicit).expanduser() if explicit else root / 'config.local.json'
    if not path.is_absolute():
        path = root / path
    config = dict(DEFAULTS)
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('配置文件必须是 JSON 对象')
        config.update(data)
    elif explicit:
        raise ValueError(f'配置文件不存在: {path}')
    for env_key, key in ENV_KEYS.items():
        if env_key in environ:
            try:
                config[key] = type(DEFAULTS[key])(environ[env_key])
            except ValueError as exc:
                raise ValueError(f'{env_key} 的值无效') from exc
    validate_config(config)
    for key in ('model_path', 'runtime_dir', 'env_prefix'):
        if config[key]:
            value = Path(config[key]).expanduser()
            config[key] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    return config
