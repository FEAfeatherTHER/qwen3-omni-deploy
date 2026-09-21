import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def checker():
    spec = importlib.util.spec_from_file_location('check_model', ROOT / 'scripts/check_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        for name in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json']:
            (self.path / name).write_text('{}')
        (self.path / 'chat_template.json').write_text(json.dumps({'chat_template': '{{ messages }}'}))
        (self.path / 'merges.txt').write_text('#version: 0.2\na b\n')
        self.index = {'metadata': {'total_size': 4}, 'weight_map': {'a': 'model-1.safetensors'}}
        self.write_index()
        header = json.dumps({'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        self.shard = self.path / 'model-1.safetensors'
        self.shard.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)

    def write_index(self):
        (self.path / 'model.safetensors.index.json').write_text(json.dumps(self.index))

    def test_valid_model_reports_shards_and_bytes(self):
        self.assertTrue((ROOT / 'scripts/check_model.py').exists(), 'checker missing')
        report = checker()(self.path)
        self.assertEqual(report['shards'], 1)
        self.assertEqual(report['tensor_bytes'], 4)

    def test_missing_shard_rejected(self):
        self.shard.unlink()
        with self.assertRaisesRegex(ValueError, 'model-1'):
            checker()(self.path)

    def test_truncated_shard_rejected(self):
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'size|truncat|offset'):
            checker()(self.path)

    def test_tensor_must_exist_in_shard(self):
        self.index['weight_map'] = {'missing': 'model-1.safetensors'}
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'missing'):
            checker()(self.path)

    def test_path_traversal_rejected(self):
        self.index['weight_map'] = {'a': '../outside.safetensors'}
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'path|outside'):
            checker()(self.path)

    def test_empty_tokenizer_rejected(self):
        (self.path / 'tokenizer_config.json').write_text('')
        with self.assertRaisesRegex(ValueError, 'tokenizer_config'):
            checker()(self.path)

    def test_missing_tokenizer_data_rejected(self):
        (self.path / 'vocab.json').unlink()
        with self.assertRaisesRegex(ValueError, 'tokenizer|vocab'):
            checker()(self.path)

    def test_missing_chat_template_rejected(self):
        (self.path / 'chat_template.json').unlink()
        with self.assertRaisesRegex(ValueError, 'chat_template'):
            checker()(self.path)

    def test_wrong_index_total_size_rejected(self):
        self.index['metadata']['total_size'] = 8
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'total_size'):
            checker()(self.path)

    def test_downloader_reuses_complete_local_model_without_network(self):
        script = ROOT / 'scripts/download_model.py'
        self.assertTrue(script.exists(), 'downloader missing')
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        result = subprocess.run(['python3', str(script), str(self.path),
                                 '--endpoint', 'http://127.0.0.1:1'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir()})

    def test_downloader_preserves_existing_mismatched_file(self):
        spec = importlib.util.spec_from_file_location('download_model', ROOT / 'scripts/download_model.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'check_model': SimpleNamespace(check_model=checker())}):
            spec.loader.exec_module(module)
        self.shard.write_bytes(b'existing user data')
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        info = SimpleNamespace(sha='a' * 40, siblings=[
            SimpleNamespace(rfilename='model-1.safetensors', size=100)])
        api = SimpleNamespace(model_info=lambda *args, **kwargs: info)
        def unexpected_download(*args, **kwargs):
            self.fail('must validate existing files before downloading')
        hub = SimpleNamespace(HfApi=lambda **kwargs: api, hf_hub_download=unexpected_download)
        with patch.dict(sys.modules, {'huggingface_hub': hub}), patch.dict('os.environ'):
            with self.assertRaisesRegex(ValueError, 'preserved'):
                module.download_model(self.path)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir()})

    def test_create_env_refuses_existing_directory(self):
        script = ROOT / 'scripts/create_env.sh'
        self.assertTrue(script.exists(), 'bootstrap missing')
        marker = self.path / 'user-file'
        marker.write_text('preserved')
        result = subprocess.run(['bash', str(script), '--prefix', str(self.path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr)
        self.assertEqual(marker.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()
