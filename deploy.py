#!/usr/bin/env python3
"""Manage the local GPU service without changing the active conda environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from qwen_deploy.config import load_config
from qwen_deploy import service

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Qwen3-Omni 本地服务管理')
    parser.add_argument('--config', help='JSON 配置路径；默认 config.local.json')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('config', '输出有效配置'), ('doctor', '检查启动条件（需空闲端口/GPU）'),
                             ('start', '在 tmux 中启动并等待就绪'), ('stop', '停止本仓库拥有的服务'),
                             ('status', '查看进程和 HTTP 健康状态')]:
        commands.add_parser(name, help=help_text)
    run_parser = commands.add_parser('run', help='前台运行；Ctrl+C 停止')
    run_parser.add_argument('--launch-id', help=argparse.SUPPRESS)
    run_parser.add_argument('--log-path', help=argparse.SUPPRESS)
    logs = commands.add_parser('logs', help='查看当前服务日志')
    logs.add_argument('--follow', action='store_true')
    logs.add_argument('--lines', type=int, default=60)
    args = parser.parse_args()
    try:
        config = load_config(ROOT, args.config)
        if args.command == 'config':
            print(json.dumps(config, ensure_ascii=False, indent=2))
        elif args.command == 'doctor':
            print(json.dumps(service.doctor(ROOT, config), ensure_ascii=False, indent=2))
        elif args.command == 'run':
            return service.run(ROOT, config, args.launch_id, args.log_path)
        elif args.command == 'start':
            return service.start(ROOT, config)
        elif args.command == 'stop':
            return service.stop(config)
        elif args.command == 'status':
            result = service.status(config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['ready'] else 1
        elif args.command == 'logs':
            state = service.read_json(Path(config['runtime_dir']) / 'server.json')
            path = state.get('log_path')
            if not path or not Path(path).is_file():
                candidates = sorted(Path(config['runtime_dir']).glob('server-*.log'))
                if not candidates:
                    raise ValueError('没有服务日志；前台 run 的输出直接显示在终端')
                path = str(candidates[-1])
            command = ['tail', '-n', str(args.lines)] + (['-f'] if args.follow else []) + [path]
            return subprocess.call(command)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
