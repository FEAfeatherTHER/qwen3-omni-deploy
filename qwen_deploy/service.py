"""Linux/tmux lifecycle for exactly one owned vLLM engine."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from .config import ENV_KEYS


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path, value):
    # Only manager-owned state files are replaced.
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


@contextmanager
def lock(runtime, name):
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / name).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('同一目录的服务操作正在运行') from exc
        yield


def resolve_env(config):
    if config['env_prefix']:
        prefix = Path(config['env_prefix'])
    else:
        conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
        if not conda:
            raise ValueError('找不到 conda；请设置 QWEN_ENV_PREFIX 或将 conda 加入 PATH')
        result = subprocess.run([conda, 'env', 'list', '--json'], capture_output=True, text=True, check=True)
        matches = [Path(p) for p in json.loads(result.stdout)['envs'] if Path(p).name == config['env_name']]
        if len(matches) != 1:
            raise ValueError(f'无法唯一定位环境 {config["env_name"]}；请设置 env_prefix')
        prefix = matches[0]
    if not (prefix / 'bin/python').is_file():
        raise ValueError(f'环境 Python 不存在: {prefix}/bin/python')
    return prefix


def build_command(config, prefix):
    return [str(prefix / 'bin/python'), '-m', 'vllm.entrypoints.openai.api_server',
            '--model', config['model_path'], '--served-model-name', config['model_name'],
            '--host', config['host'], '--port', str(config['port']), '--dtype', 'bfloat16',
            '--tensor-parallel-size', str(config['tensor_parallel_size']),
            '--max-model-len', str(config['max_model_len']),
            '--max-num-seqs', str(config['max_num_seqs']),
            '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
            '--limit-mm-per-prompt', json.dumps({'audio': config['max_audio'], 'image': 1, 'video': 0}),
            '--enforce-eager']


def check_port(host, port):
    with socket.socket() as sock:
        # Match the HTTP server: completed connections in TIME_WAIT do not
        # prevent a restart, while an existing listening socket still does.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise ValueError(f'端口 {host}:{port} 不可用；现有服务不会被停止') from exc


def check_gpus(config):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    available = {}
    for row in result.stdout.splitlines():
        index, total, free = [int(value.strip()) for value in row.split(',')]
        available[index] = {'total_mib': total, 'free_mib': free}
    selected = {}
    for gpu in map(int, config['gpus'].split(',')):
        if gpu not in available:
            raise ValueError(f'GPU {gpu} 不存在')
        info = available[gpu]
        minimum = info['total_mib'] * config['gpu_memory_utilization']
        if info['free_mib'] < minimum:
            raise ValueError(f'GPU {gpu} 空闲显存不足: {info["free_mib"]} MiB，需要至少 {minimum:.0f} MiB')
        selected[gpu] = info
    return selected


def doctor(root, config):
    prefix = resolve_env(config)
    subprocess.run([str(prefix / 'bin/python'), '-c',
                    'from importlib.metadata import version; '
                    'print({p:version(p) for p in ["vllm","torch","transformers","qwen-omni-utils"]})'], check=True)
    subprocess.run([str(prefix / 'bin/python'), str(root / 'scripts/check_model.py'), config['model_path']], check=True)
    check_port(config['host'], config['port'])
    gpu_report = check_gpus(config)
    return {'environment': str(prefix), 'gpus': gpu_report, 'command': build_command(config, prefix)}


def process_identity(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if data[0] == 'Z':
            return {}
        return {'pid': pid, 'start_ticks': data[19], 'pgid': int(data[2])}
    except (OSError, IndexError, ValueError):
        return {}


def process_matches(identity):
    return bool(identity) and process_identity(identity.get('pid', -1)) == identity


def owned_processes(identity, known=()):
    """Only discover a group while a recorded process proves its ownership."""
    if not identity or identity.get('pid') != identity.get('pgid'):
        return []
    anchors = [identity, *known]
    if not any(member.get('pgid') == identity['pgid'] and process_matches(member) for member in anchors):
        return []
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            member = process_identity(int(path.name))
            if (member and member['pgid'] == identity['pgid']
                    and int(member['start_ticks']) >= int(identity['start_ticks'])):
                members.append(member)
    return sorted(members, key=lambda member: member['pid'])


def stop_owned(identity, timeout=60, members=()):
    owned = owned_processes(identity, members)
    if not owned:
        return False
    if identity['pgid'] != identity['pid']:
        raise ValueError('进程组不属于该服务，拒绝停止')
    try:
        os.killpg(identity['pgid'], signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + timeout
    while owned and time.monotonic() < deadline:
        time.sleep(0.1)
        owned = owned_processes(identity, owned)
    if owned:
        try:
            os.killpg(identity['pgid'], signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while owned and time.monotonic() < deadline:
            time.sleep(0.1)
            owned = owned_processes(identity, owned)
        if owned:
            raise ValueError('仍有服务工作进程未退出，不能报告停止成功')
    return True


def api_ready(config):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        base = f'http://{config["host"]}:{config["port"]}'
        with opener.open(base + '/health', timeout=2) as response:
            if response.status != 200:
                return False
        with opener.open(base + '/v1/models', timeout=2) as response:
            return any(model.get('id') == config['model_name'] for model in json.load(response)['data'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def status(config):
    state = read_json(Path(config['runtime_dir']) / 'server.json')
    engine = state.get('engine', {})
    members = owned_processes(engine, state.get('members', []))
    # Readiness belongs to the running instance, not a caller's changed config.
    endpoint = state.get('config', config)
    return {**state, 'running': bool(members), 'ready': process_matches(engine) and api_ready(endpoint)}


def run(root, config, launch_id=None, log_path=None):
    runtime = Path(config['runtime_dir'])
    with lock(runtime, 'service.lock'):
        report = doctor(root, config)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=config['gpus'], VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                   NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1')
        env.setdefault('OMP_NUM_THREADS', '4')
        def shutdown(signum, frame):
            raise KeyboardInterrupt

        previous = {}
        child, identity, members, state = None, {}, [], {}
        try:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous[sig] = signal.signal(sig, shutdown)
            child = subprocess.Popen(report['command'], env=env, start_new_session=True, cwd=root)
            identity = process_identity(child.pid)
            state = {'launch_id': launch_id or uuid.uuid4().hex, 'engine': identity,
                     'runner': process_identity(os.getpid()), 'config': config,
                     'started_at': datetime.now(timezone.utc).isoformat(), 'log_path': log_path,
                     'command': report['command']}
            write_json(runtime / 'server.json', state)
            while True:
                members = owned_processes(identity, members)
                if members != state.get('members'):
                    state['members'] = members
                    write_json(runtime / 'server.json', state)
                try:
                    code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            code = 0
        finally:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if child is not None:
                    stop_owned(identity, members=members)
                    child.wait()
                    if state:
                        write_json(runtime / 'server.json', {**state, 'exit_code': child.returncode,
                                                            'stopped_at': datetime.now(timezone.utc).isoformat()})
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        return code


def start(root, config):
    runtime = Path(config['runtime_dir'])
    with lock(runtime, 'control.lock'):
        if status(config)['running']:
            raise ValueError('该仓库已有运行中的服务；使用 status 查看')
        if not shutil.which('tmux'):
            raise ValueError('找不到 tmux；可以使用 run 前台运行')
        session = config['tmux_session']
        exists = subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode == 0
        if exists:
            raise ValueError(f'tmux 会话 {session} 已存在；请设置其他 tmux_session 或检查原会话')
        doctor(root, config)
        launch_id = uuid.uuid4().hex
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + launch_id[:8]
        snapshot = runtime / f'config-{stamp}.json'
        log_path = runtime / f'server-{stamp}.log'
        effective = dict(config, env_prefix=str(resolve_env(config)))
        write_json(snapshot, effective)
        # tmux can have stale environment variables from an older login.
        clean_env = ['env']
        for name in [*ENV_KEYS, 'QWEN_CONFIG']:
            clean_env.extend(['-u', name])
        args = clean_env + [sys.executable, str(root / 'deploy.py'), '--config', str(snapshot),
                            'run', '--launch-id', launch_id, '--log-path', str(log_path)]
        command = 'exec ' + shlex.join(args) + ' >> ' + shlex.quote(str(log_path)) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(root), command], check=True)
        print(f'服务启动中，日志: {log_path}', flush=True)
        deadline = time.monotonic() + config['startup_timeout']
        while time.monotonic() < deadline:
            state = status(effective)
            if state.get('launch_id') == launch_id:
                if state['ready']:
                    print(f'服务就绪: http://127.0.0.1:{config["port"]}/v1', flush=True)
                    return 0
                if not state['running']:
                    raise ValueError(f'服务启动失败，请查看 {log_path}')
            if subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode != 0:
                raise ValueError(f'启动进程已退出，请查看 {log_path}')
            time.sleep(2)
        raise ValueError(f'等待就绪超时；服务可能仍在加载，用 status 检查，日志: {log_path}')


def stop(config):
    runtime = Path(config['runtime_dir'])
    with lock(runtime, 'control.lock'):
        state = read_json(runtime / 'server.json')
        engine = state.get('engine', {})
        members = owned_processes(engine, state.get('members', []))
        if not members:
            print('没有该仓库拥有的运行中服务')
            return 0
        runner = state.get('runner', {})
        if process_matches(runner):
            os.kill(runner['pid'], signal.SIGTERM)
            deadline = time.monotonic() + 70
            while process_matches(runner) and time.monotonic() < deadline:
                time.sleep(0.2)
        stop_owned(engine, members=members)
        if owned_processes(engine, members):
            raise ValueError('服务尚未退出，请检查日志')
        print('服务已停止')
        return 0
