import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from qwen_deploy.config import load_config, validate_config
from qwen_deploy.service import build_command, check_port, process_matches, process_identity, stop_owned
from qwen_deploy import service


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen test ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_local_config_and_environment_precedence(self):
        (self.root / 'config.local.json').write_text(json.dumps({'port': 9000, 'gpus': '0,1,3,4'}))
        config = load_config(self.root, environ={'QWEN_PORT': '9001'})
        self.assertEqual(config['port'], 9001)
        self.assertEqual(config['gpus'], '0,1,3,4')
        self.assertEqual(config['tensor_parallel_size'], 4)

    def test_relative_paths_follow_relocated_repo(self):
        (self.root / 'config.local.json').write_text(json.dumps({'model_path': 'weights/model'}))
        config = load_config(self.root, environ={})
        self.assertEqual(config['model_path'], str(self.root / 'weights/model'))
        self.assertEqual(config['runtime_dir'], str(self.root / '.runtime'))

    def test_invalid_gpu_port_or_unknown_config_is_rejected(self):
        cases = [{'gpus': '0,0,1,2'}, {'gpus': '0,1'}, {'port': 0}, {'port': 65536},
                 {'host': '0.0.0.0'}, {'max_num_seq': 4}, {'gpu_memory_utilization': 1.1}]
        for update in cases:
            with self.subTest(update=update):
                path = self.root / 'config.local.json'
                path.write_text(json.dumps(update))
                with self.assertRaises(ValueError):
                    load_config(self.root, environ={})

    def test_command_preserves_paths_and_uses_four_gpu_thinker(self):
        config = load_config(self.root, environ={})
        command = build_command(config, Path('/tmp/conda env'))
        self.assertEqual(command[:3], ['/tmp/conda env/bin/python', '-m', 'vllm.entrypoints.openai.api_server'])
        self.assertEqual(command[command.index('--tensor-parallel-size')+1], '4')
        self.assertEqual(json.loads(command[command.index('--limit-mm-per-prompt')+1]), {'audio': 4, 'image': 1, 'video': 0})
        self.assertNotIn('--allowed-local-media-path', command)


class LifecycleTests(unittest.TestCase):
    def test_failed_state_write_cleans_up_spawned_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load_config(Path(folder), environ={})
            children = []
            original = subprocess.Popen

            def spawn(*args, **kwargs):
                process = original(*args, **kwargs)
                children.append(process)
                return process

            try:
                with patch.object(service, 'doctor', return_value={'command': [sys.executable, '-c', 'import time; time.sleep(60)']}), \
                     patch.object(service.subprocess, 'Popen', side_effect=spawn), \
                     patch.object(service, 'write_json', side_effect=OSError('No space left on device')):
                    with self.assertRaises(OSError):
                        service.run(Path(folder), config)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll(), 'Failed startup must reap its engine')
            finally:
                for process in children:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)

    def check_orphan_cleanup(self, leader_exits_first):
        with tempfile.TemporaryDirectory() as folder:
            worker_path = Path(folder) / 'worker.pid'
            worker_code = ('import os,signal,time; from pathlib import Path; '
                           'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                           f'Path({str(worker_path)!r}).write_text(str(os.getpid())); time.sleep(60)')
            leader_code = ('import subprocess,sys,time; '
                           f'subprocess.Popen([sys.executable,"-c",{worker_code!r}]); time.sleep(60)')
            leader = subprocess.Popen([sys.executable, '-c', leader_code], start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not worker_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                worker = process_identity(int(worker_path.read_text()))
                identity = process_identity(leader.pid)
                if leader_exits_first:
                    leader.terminate()
                    leader.wait(timeout=5)
                    self.assertTrue(service.owned_processes(identity, [worker]))
                self.assertTrue(stop_owned(identity, timeout=0.3, members=[worker]))
                self.assertFalse(process_matches(worker), 'Worker must not remain after stop succeeds')
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait(timeout=5)

    def test_worker_that_ignores_term_is_removed_after_leader_exits(self):
        self.check_orphan_cleanup(False)

    def test_recorded_worker_is_cleaned_up_even_if_leader_already_crashed(self):
        self.check_orphan_cleanup(True)

    def test_occupied_port_is_rejected(self):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            with self.assertRaisesRegex(ValueError, '端口'):
                check_port('127.0.0.1', sock.getsockname()[1])

    def test_recently_closed_server_port_can_be_reused(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            listener.listen()
            client = socket.create_connection(('127.0.0.1', port))
            connection, _ = listener.accept()
            connection.shutdown(socket.SHUT_WR)
            self.assertEqual(client.recv(1), b'')
            connection.close()
            client.close()
        check_port('127.0.0.1', port)

    def test_stale_process_identity_does_not_terminate_process(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        self.addCleanup(lambda: process.poll() is None and process.terminate())
        self.addCleanup(lambda: None)
        identity = process_identity(process.pid)
        self.assertTrue(process_matches(identity))
        stale = dict(identity, start_ticks=str(int(identity['start_ticks']) + 1))
        self.assertFalse(process_matches(stale))
        self.assertFalse(stop_owned(stale, timeout=0.1))
        self.assertIsNone(process.poll())
        self.assertTrue(stop_owned(identity, timeout=5))
        process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
